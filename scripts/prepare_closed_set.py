# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""Build the closed-set manifests for MMLU and MMMU.

    python scripts/prepare_closed_set.py --data-root /path/to/data [--only mmlu,mmmu]

Writes `<data-root>/closed_set/{mmlu,mmmu}/<task>.jsonl` (+ images). Every question
appears once.
"""
import argparse
import ast
import collections
import json
import os
import re

MC_INSTRUCTION = "Answer the multiple-choice question by selecting the correct option."
PLACEHOLDER = re.compile(r"<\s*image\s*\d*\s*>")

MMLU_REPO, MMLU_REVISION = "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe"
MMMU_REPO, MMMU_REVISION = "MMMU/MMMU", "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"


def option_label(index):
    label, index = "", index + 1
    while index:
        index, rem = divmod(index - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def render_options(query, options):
    lines = ["%s. %s" % (option_label(i), o) for i, o in enumerate(options)]
    return (query or "") + "\nOptions:\n" + "\n".join(lines)


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dedup_options(opts):
    out, seen = [], set()
    for i, o in enumerate(opts):
        o = str(o).strip() or "option %d" % (i + 1)
        if o in seen:
            o = "%s: %s" % (chr(ord("A") + i), o)
        seen.add(o)
        out.append(o)
    return out


# ---------------------------------------------------------------- MMLU (test split, 57 subjects)
def build_mmlu(out):
    from datasets import get_dataset_config_names, load_dataset

    subjects = [s for s in get_dataset_config_names(MMLU_REPO) if s not in ("all", "auxiliary_train")]
    total = 0
    for subject in sorted(subjects):
        ds = load_dataset(MMLU_REPO, subject, split="test", revision=MMLU_REVISION)
        rows = []
        for i, r in enumerate(ds):
            options = _dedup_options(r["choices"])
            query = "%s\nquestion: %s" % (MC_INSTRUCTION, r["question"])
            rows.append({"id": "mmlu/%s/%d" % (subject, i), "source": "mmluF_" + subject,
                         "query": render_options(query, options), "answer": options[r["answer"]],
                         "options": options, "k": len(options)})
        write(os.path.join(out, "mmluF_%s.jsonl" % subject), rows)
        total += len(rows)
    print("[mmlu] %d subjects, %d questions" % (len(subjects), total))


# ---------------------------------------------------------------- MMMU (validation + test)
def _letter_to_option(ans, opts):
    a = str(ans).strip()
    if len(a) == 1 and a.upper().isalpha():
        i = ord(a.upper()) - ord("A")
        if 0 <= i < len(opts):
            return opts[i]
    return a if a in opts else None


def build_mmmu(out):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f.rfilename for f in HfApi().repo_info(MMMU_REPO, repo_type="dataset", revision=MMMU_REVISION).siblings
                   if f.rfilename.endswith(".parquet") and ("/validation-" in f.rfilename or "/test-" in f.rfilename))
    buckets, stats = collections.defaultdict(list), collections.Counter()
    for fname in files:
        subject = re.sub(r"[^0-9A-Za-z]+", "_", fname.split("/")[0]).strip("_")
        local = hf_hub_download(MMMU_REPO, fname, repo_type="dataset", revision=MMMU_REVISION)
        for r in pq.read_table(local).to_pylist():
            if (r.get("question_type") or "multiple-choice") != "multiple-choice":
                stats["open"] += 1
                continue
            raw = r["options"]
            try:
                opts = ast.literal_eval(raw) if isinstance(raw, str) else list(raw or [])
            except (ValueError, SyntaxError):
                stats["bad_options"] += 1
                continue
            if not isinstance(opts, (list, tuple)) or len(opts) < 2:
                stats["bad_options"] += 1
                continue
            opts = _dedup_options(opts)
            gold = _letter_to_option(r.get("answer"), opts)
            if gold is not None:
                gold = PLACEHOLDER.sub("image", gold)
            # Image placeholders become the word "image". Questions whose options are then no
            # longer distinguishable (they differed only by which image they point at) cannot
            # be scored with text candidates.
            opts = [PLACEHOLDER.sub("image", o) for o in opts]
            if len(set(opts)) < len(opts):
                stats["image_options"] += 1
                continue
            if gold is None:
                stats["no_gold"] += 1
                continue
            images = []
            for i in range(1, 8):
                d = r.get("image_%d" % i)
                if d and d.get("bytes"):
                    rel = "images/%s_%d.png" % (r["id"], i)
                    path = os.path.join(out, rel)
                    if not os.path.exists(path):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "wb") as f:
                            f.write(d["bytes"])
                    images.append(rel)
            query = PLACEHOLDER.sub("image", r.get("question") or "")
            rec = {"id": "mmmu:%s" % r["id"], "source": "mmmu_" + subject,
                   "query": render_options(query, opts), "answer": gold, "options": opts, "k": len(opts)}
            if images:
                rec["images"] = images
            buckets[subject].append(rec)
    for subject, rows in sorted(buckets.items()):
        write(os.path.join(out, "mmmuctx_%s.jsonl" % subject), rows)
    print("[mmmu] %d subjects, %d questions, skipped %s"
          % (len(buckets), sum(map(len, buckets.values())), dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--only", default="mmlu,mmmu")
    args = ap.parse_args()
    builders = {"mmlu": build_mmlu, "mmmu": build_mmmu}
    for name in args.only.split(","):
        builders[name](os.path.join(args.data_root, "closed_set", name))


if __name__ == "__main__":
    main()
