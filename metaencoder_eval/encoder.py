# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""Encode parser rows with the public `MetaEncoder`.

The task parsers emit rows in the training repo's wire format (see `prompt.py`). `to_item`
converts one (text, media) pair into a `MetaEncoder` item using the same routing rules the
training-repo collator applied:

- `<|patch|>` / `<|video|>` markers are dropped from the text; the text before `\\ue000` is the
  instruction and the text after it is the content.
- Frames in a row's `query_video` are a video. Otherwise a row's media list is a video when its
  text carries `<|video|>`, and independent images when it does not.
- Images are passed through unresized; the processor's `max_image_tokens` does the resizing.
"""
import io
import os
import pickle
import sys

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from metaencoder_eval.prompt import INSTRUCTION_SEP, VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS, MUSE_GLIMMER

IMAGE_TOKEN = VLM_IMAGE_TOKENS[MUSE_GLIMMER]
VIDEO_TOKEN = VLM_VIDEO_TOKENS[MUSE_GLIMMER]


def _load_images(media):
    if not isinstance(media, dict):
        return None
    paths = media.get("paths") or []
    blobs = media.get("bytes") or [None] * len(paths)
    images = []
    for path, blob in zip(paths, blobs):
        if blob is not None:
            im = Image.open(io.BytesIO(blob))
        elif path is not None:
            im = Image.open(path)
        else:
            continue
        im.load()
        images.append(im)
    return images or None


def to_item(text, media, video_media=None):
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    images = _load_images(media)
    video = _load_images(video_media)
    if video is None and images and VIDEO_TOKEN in text:
        video, images = images, None

    clean = " ".join(text.replace(IMAGE_TOKEN, " ").replace(VIDEO_TOKEN, " ").split())
    item = {}
    if INSTRUCTION_SEP in clean:
        instruction, content = (p.strip() for p in clean.split(INSTRUCTION_SEP, 1))
        item["text"] = content
        if instruction:
            item["instruction"] = instruction
    else:
        item["text"] = clean
    if images:
        item["images"] = images
    if video:
        item["video"] = video
    return item


def flatten_rows(rows, side):
    """Parser rows -> (texts, media, videos) per embedding. Candidate rows can carry several entries."""
    tkey, mkey, vkey = ("query_text", "query_image", "query_video") if side == "qry" else ("cand_text", "cand_image", None)
    texts, media, videos = [], [], []
    for row in rows:
        ts, ms = row[tkey], row[mkey]
        vs = row.get(vkey) if vkey else None
        if not isinstance(ts, list):
            ts, ms = [ts], [ms]
        ms = ms if isinstance(ms, list) else [ms] * len(ts)
        vs = vs if isinstance(vs, list) else [None] * len(ts)
        for t, m, v in zip(ts, ms, vs):
            texts.append(t)
            media.append(m)
            videos.append(v)
    return texts, media, videos


class DistributedEncoder:
    """Shards a list of inputs contiguously across ranks, like `split_dataset_by_node`."""

    def __init__(self, model_path, batch_size=8, gather_dir=None):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.batch_size = batch_size
        self.gather_dir = gather_dir
        sys.path.insert(0, model_path)
        from modeling_metaencoder import MetaEncoder

        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.model = MetaEncoder.from_pretrained(model_path, device_map={"": device})

    def encode(self, texts, media, videos, tag):
        n = len(texts)
        padded = n + (-n) % self.world
        idx = [i % n for i in range(padded)] if n else []
        shard = padded // self.world
        mine = idx[self.rank * shard:(self.rank + 1) * shard]
        out = []
        for s in range(0, len(mine), self.batch_size):
            items = [to_item(texts[i], media[i], videos[i]) for i in mine[s:s + self.batch_size]]
            out.append(self.model.encode(items, batch_size=len(items)).float().cpu().numpy())
        local = np.concatenate(out) if out else np.zeros((0, self.model.embedding_dim), np.float32)
        return self._gather(local, tag)[:n]

    def _gather(self, local, tag):
        if self.world == 1:
            return local
        os.makedirs(self.gather_dir, exist_ok=True)
        path = os.path.join(self.gather_dir, f"{tag}.r{self.rank}.pkl")
        with open(path, "wb") as f:
            pickle.dump(local, f)
        dist.barrier()
        parts = []
        for r in range(self.world):
            with open(os.path.join(self.gather_dir, f"{tag}.r{r}.pkl"), "rb") as f:
                parts.append(pickle.load(f))
        dist.barrier()
        os.remove(path)
        return np.concatenate(parts)
