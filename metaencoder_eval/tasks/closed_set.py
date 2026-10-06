# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""Closed-set multiple-choice manifests (MMLU, MMMU, ImaJEV).

One JSONL row per question:

    {"id": ..., "query": ..., "answer": <gold option text>, "options": [<option text>, ...],
     "images": [<path relative to the manifest>, ...]}   # images optional

The query is "{query_instruction} {query}" with the images attached; the candidates are the
row's own option texts and the score is hit@1 among them. Candidate order is gold first, then
the remaining options, as in the reported runs (it only matters for exact ties).
"""
import json
import os

from datasets import Dataset

from metaencoder_eval.prompt import VLM_IMAGE_TOKENS, MUSE_GLIMMER, process_input_text
from metaencoder_eval.tasks.base_eval_dataset import AutoEvalPairDataset

DATASET_PARSER_NAME = "closed_set"
MAX_IMAGES = 8


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_closed_set_dataset(model_args, data_args, *args, **kwargs):
    path = kwargs["dataset_path"]
    instruction = kwargs.get("query_instruction", "Select the correct option.")
    base = os.path.dirname(os.path.abspath(path))

    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        gold = (r.get("answer") or "").strip()
        query = (r.get("query") or "").strip()
        options = [str(o).strip() for o in (r.get("options") or []) if str(o).strip()]
        if not gold or not query:
            continue
        declared = list(r.get("images") or [])
        paths = [p if os.path.isabs(p) else os.path.join(base, p) for p in declared if p][:MAX_IMAGES]
        if declared and not paths:
            continue

        text = process_input_text(instruction, MUSE_GLIMMER, text=query, add_image_token=bool(paths))
        if len(paths) > 1:
            text = (VLM_IMAGE_TOKENS[MUSE_GLIMMER] + " ") * (len(paths) - 1) + text
        media = {"bytes": [None] * len(paths), "paths": paths, "resolutions": [None] * len(paths)} if paths else None
        names = [gold] + [o for o in options if o != gold]
        rows.append({
            "query_text": [text],
            "query_image": [media],
            "cand_text": names,
            "cand_image": [None] * len(names),
            "dataset_infos": {"id": str(r.get("id")), "cand_names": names, "label_name": gold},
        })
    return Dataset.from_list(rows), None
