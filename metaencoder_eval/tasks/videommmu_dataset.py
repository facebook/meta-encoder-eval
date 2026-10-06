# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import cv2
import datasets
from datasets import load_dataset

from metaencoder_eval.tasks.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from metaencoder_eval.prompt import process_input_text
from metaencoder_eval.utils.dataset_utils import sample_dataset
from metaencoder_eval.utils.vision_utils import load_frames, process_video_frames, qa_template

TASK_PROMPT = (
    "Given a video and a question, select the most accurate answer from the provided "
    "candidates. Return only the exact text of your chosen answer. Question: "
)

DATASET_PARSER_NAME = "videommmu"
DATASET_HF_PATH = "lmms-eval/VideoMMMU"
DATASET_REVISION = "d1c35ac933123d79e877b7f1b9506afb0309cf1b"
SUBSET_NAMES = ["Perception", "Comprehension", "Adaptation"]


def _extract_frames(video_path, frame_dir, max_frames_saved):
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames_saved)
    idx = saved = 0
    while saved < max_frames_saved:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{saved:04d}.jpeg"), frame)
        saved += 1
        idx += step
    cap.release()
    return saved


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs["model_backbone"]
    max_frames_saved = kwargs["max_frames_saved"]
    video_root = kwargs["video_root"]
    frame_root = kwargs["frame_root"]
    num_frames = kwargs["num_frames"]

    query_texts, query_images, query_videos, cand_texts, cand_images, dataset_infos = [], [], [], [], [], []
    n = len(batch_dict["id"])

    for i in range(n):
        qtype = batch_dict["question_type"][i]
        if qtype != "multiple-choice":
            continue
        options = list(batch_dict["options"][i] or [])
        answer = batch_dict["answer"][i]
        if len(options) < 2 or not isinstance(answer, str) or len(answer) != 1:
            continue
        # `answer` is a LETTER ("A".."J"), not an index and not the option text.
        # qa_template matches gold by option text, so resolve the letter first.
        ai = ord(answer.upper()) - ord("A")
        if not (0 <= ai < len(options)):
            continue
        answer_text = options[ai]

        video_id = batch_dict["id"][i]
        subset = batch_dict["subset"][i]
        # Adaptation questions carry a still image; the query gets the clip's frames as a video
        # and the still as a separate image, as in the official protocol.
        still = None
        im = (batch_dict.get("image") or [None] * n)[i]
        if im:
            os.makedirs(f"{frame_root}/_stills", exist_ok=True)
            sp = f"{frame_root}/_stills/{batch_dict['id'][i]}.png"
            if not os.path.exists(sp):
                b = im["bytes"] if isinstance(im, dict) else getattr(im, "_bytes", None)
                if b:
                    # re-encode to RGB: a chunk of these stills are RGBA, and a 4-channel
                    # array makes the image processor raise "Unable to infer channel
                    # dimension format".
                    from PIL import Image as _PILImage
                    import io as _io
                    _PILImage.open(_io.BytesIO(b)).convert("RGB").save(sp)
            if os.path.exists(sp):
                still = sp
        qtext = re.sub(r"<\s*image\s*\d*\s*>", "image", batch_dict["question"][i] or "")
        query = process_input_text(TASK_PROMPT, model_backbone,
                                   text=qtext, add_video_token=True,
                                   add_image_token=bool(still))
        query, cands, gold, gold_idx = qa_template(query, options, answer_text)

        video_path = f"{video_root}/{subset}/{video_id}.mp4"
        frame_dir = f"{frame_root}/{subset}/{video_id}"
        frames = load_frames(frame_dir)
        if not frames:
            if not os.path.exists(video_path):
                continue
            _extract_frames(video_path, frame_dir, max_frames_saved)
        paths = process_video_frames(frame_dir, num_frames=num_frames)
        if not paths:
            continue

        query_texts.append([query])
        query_videos.append([{"bytes": [None] * len(paths), "paths": paths,
                              "resolutions": [None] * len(paths)}])
        query_images.append([{"bytes": [None], "paths": [still], "resolutions": [None]}] if still else [None])
        cand_texts.append(cands)
        cand_images.append([None] * len(cands))
        dataset_infos.append({
            "video_id": video_id, "subset": subset, "qa_type": batch_dict["qa_type"][i],
            "query": query, "cand_names": cands, "label_name": gold,
            "answer": gold, "answer_idx": gold_idx, "qry_frame_paths": paths,
        })

    return {"query_text": query_texts, "query_image": query_images, "query_video": query_videos,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_videommmu_dataset(model_args, data_args, *args, **kwargs):
    subsets = []
    only = kwargs.get("subset") or None
    for name in ([only] if only else SUBSET_NAMES):
        ds = load_dataset(DATASET_HF_PATH, name, split="test", revision=DATASET_REVISION)
        ds = ds.add_column("subset", [name] * len(ds))
        if "image" not in ds.column_names:
            ds = ds.add_column("image", [None] * len(ds))
        ds = ds.cast_column("image", datasets.Image(decode=False))
        subsets.append(ds)
    dataset = datasets.concatenate_datasets(subsets)
    print(f"Loading {DATASET_HF_PATH}, {len(dataset)} samples")

    kwargs["dataset_name"] = DATASET_PARSER_NAME
    kwargs["model_backbone"] = model_args.model_backbone
    kwargs["image_resolution"] = data_args.image_resolution
    kwargs["global_dataset_name"] = kwargs.get("global_dataset_name", DATASET_PARSER_NAME)
    # data_prepare drops rows (non-MC, unresolvable gold, missing frames). With
    # batched=True the retained source columns keep the ORIGINAL row count, so arrow
    # fails with "expected length 64 but got length 60". Filter up front and drop the
    # source columns so only the emitted ones define the batch length.
    dataset = dataset.filter(lambda r: r["question_type"] == "multiple-choice")
    print(f"multiple-choice rows: {len(dataset)}")
    dataset = sample_dataset(dataset, **kwargs)
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=64, num_proc=4,
                          remove_columns=dataset.column_names,
                          drop_last_batch=False, load_from_cache_file=False)
    return dataset, None
