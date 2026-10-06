# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""JEVBench: accuracy over each row's own label set.

    python -m metaencoder_eval.jevbench --model /path/to/meta-encoder \
        --data-dir /path/to/jevbench/datasets/public --out results/jevbench

The query renders the row's state, instruction and label criteria; the candidates are the bare
labels, which point back into the criteria block.
"""
import argparse
import json
import os
import sys

import torch


def render_criteria(criteria):
    if isinstance(criteria, dict):
        return "\n".join("- %s: %s" % (k, v) for k, v in criteria.items())
    if isinstance(criteria, (list, tuple)):
        return "\n".join("- %d: %s" % (i, v) for i, v in enumerate(criteria))
    return str(criteria or "")


def build_query(row):
    q = row["question"] if isinstance(row["question"], dict) else {"instructions": row["question"]}
    parts = []
    if row.get("state"):
        parts += [str(row["state"]).strip(), ""]
    if q.get("instructions"):
        parts.append("Task: " + str(q["instructions"]).strip())
    crit = render_criteria(q.get("criteria"))
    if crit:
        parts += ["Criteria:", crit]
    if row["labels"]:
        parts.append("Options: " + ", ".join(str(o) for o in row["labels"]))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="original,easy,hard")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, args.model)
    from modeling_metaencoder import MetaEncoder
    model = MetaEncoder.from_pretrained(args.model)

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for split in args.splits.split(","):
        rows = [json.loads(l) for l in open(os.path.join(args.data_dir, f"{split}.jsonl")) if l.strip()]
        qv = model.encode([{"text": build_query(r)} for r in rows], batch_size=args.batch_size).float().cpu()
        preds = []
        for i, r in enumerate(rows):
            cv = model.encode([{"text": str(l)} for l in r["labels"]], batch_size=args.batch_size).float().cpu()
            sims = cv @ qv[i]
            pred = r["labels"][int(torch.argmax(sims))]
            preds.append({"id": r.get("id"), "pred": pred, "expected": r["expected"],
                          "hit": str(pred) == str(r["expected"]), "scores": sims.tolist()})
        acc = sum(p["hit"] for p in preds) / len(preds)
        summary[split] = {"accuracy": acc, "n": len(preds)}
        with open(os.path.join(args.out, f"{split}_pred.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"[jevbench/{split}] accuracy={acc:.4f} n={len(preds)}", flush=True)
    json.dump({"suite": "jevbench", "metric": "accuracy", "splits": summary},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
