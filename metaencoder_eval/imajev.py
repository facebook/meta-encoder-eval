# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""ImajevBench v2.0-lite in the benchmark's own run format.

    python -m metaencoder_eval.imajev --model /path/to/meta-encoder \
        --dataset /path/to/data/imajev-bench --harness /path/to/data/imajev-harness/src --out results/imajev

For each split (dev, calibration, test) and condition (full, no_image, no_state) this writes a run
folder (`manifest.json`, `predictions.jsonl`, `raw.jsonl`, `completion.json`) that the harness's
`imajev_bench score` verifies and scores. Dev and calibration are scored here; the test split has
its labels withheld, so its run folder is what gets submitted to the maintainers.

Interface: one task embedding per item and one embedding per candidate. The task is the question,
the state and the images, followed by the allowed answers as a lettered list that ends with the
harness's neutral Unknown option. The candidates are the bare answer labels (`yes`, `no`, each
option or level value, `unknown`). The answer is the highest-cosine candidate; choosing `unknown`
is an abstention. This encoding was chosen on the dev and calibration splits. Probabilities are a
softmax over the cosines with one temperature fitted on the calibration split.
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
from PIL import Image

INSTRUCTION = "Select the correct option."
UNKNOWN_KEY = "__unknown__"
# imajev_bench.harness.NEUTRAL_UNKNOWN
UNKNOWN_TEXT = "unknown — the supplied images and state do not determine the answer"
TEMPERATURES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
SPLITS = ("dev", "calibration", "test")
CONDITIONS = ("full", "no_image", "no_state")


def render_state(state):
    if not isinstance(state, dict):
        return str(state or "")[:4000]
    return "\n".join(f"{k}: {v}" for k, v in state.items() if v is not None)[:4000]


def candidates(field):
    """(probability key, typed value, candidate label, option line for the prompt), Unknown last."""
    if field["type"] == "boolean":
        answers = [("true", True, "yes", field.get("yes_description")),
                   ("false", False, "no", field.get("no_description"))]
    else:
        items = field.get("options") if field["type"] == "choice" else field.get("levels")
        answers = [(str(o["value"]), o["value"], str(o["value"]), o.get("description")) for o in items]
    out = [(key, value, label, f"{label}: {desc}" if desc else label) for key, value, label, desc in answers]
    return out + [(UNKNOWN_KEY, None, "unknown", UNKNOWN_TEXT)]


def task_item(payload, root, condition):
    request = payload["request"]
    field = request["fields"][0]
    state = "" if condition == "no_state" else render_state(request.get("state"))
    question = field.get("question") or ""
    text = f"{question}\n\n{state}" if state else question
    lines = [c[3] for c in candidates(field)]
    text += "\nOptions:\n" + "\n".join(f"{chr(ord('A') + i)}. {line}" for i, line in enumerate(lines))
    item = {"instruction": INSTRUCTION, "text": text}
    if condition != "no_image" and payload["images"]:
        item["images"] = [Image.open(os.path.join(root, im["path"])) for im in payload["images"]]
    return item


def softmax(scores, temperature):
    return torch.softmax(torch.tensor(scores) / temperature, dim=0).tolist()


def fit_temperature(rows):
    """NLL-minimising temperature over (cosines, gold index) pairs."""
    def nll(t):
        return -sum(math.log(max(softmax(s, t)[g], 1e-15)) for s, g in rows) / len(rows)
    return min(TEMPERATURES, key=nll)


def encode_split(model, records, root, condition, batch_size):
    from imajev_bench.schema import model_payload

    payloads = [model_payload(r) for r in records]
    fields = [p["request"]["fields"][0] for p in payloads]
    cands = [candidates(f) for f in fields]
    texts = sorted({c[2] for cs in cands for c in cs})  # bare labels
    cand_emb = dict(zip(texts, model.encode([{"text": t} for t in texts], batch_size=16).float().cpu()))
    scores = []
    for s in range(0, len(payloads), batch_size):
        items = [task_item(p, root, condition) for p in payloads[s:s + batch_size]]
        q = model.encode(items, batch_size=len(items)).float().cpu()
        for qi, cs in zip(q, cands[s:s + batch_size]):
            scores.append([float(cand_emb[c[2]] @ qi) for c in cs])
    return payloads, cands, scores


def write_run(out, records, payloads, cands, scores, temperature, condition, model_path):
    from imajev_bench.runner import canonical_bytes, digest, file_digest, run_hashes

    os.makedirs(out, exist_ok=False)
    manifest = {
        "format_version": "0.1.0", "adapter": "metaencoder-embedding-match", **run_hashes(records),
        "record_count": len(records), "reviewed": all(r["annotation_status"] == "reviewed" for r in records),
        "started_at": datetime.now(timezone.utc).isoformat(), "condition": condition, "concurrency": 1,
        "model": "facebook/meta-encoder", "model_path": os.path.abspath(model_path), "precision": "bfloat16",
        "interface": "embedding match: argmax cosine(task, candidate) over the allowed answers plus Unknown; "
                     "single pass, options listed once in their record order (no rotations)",
        "task_prompt": f"{INSTRUCTION} {{question}}\\n\\n{{state as 'key: value' lines}}\\nOptions:\\n"
                       "A. {label}[: {description}]\\n... {last letter}. " + UNKNOWN_TEXT + " + images",
        "candidates": "bare answer labels: yes / no / option or level value / unknown",
        "encoding_selection": "chosen on the dev and calibration splits",
        "calibration": {"method": "softmax temperature fitted by NLL on the calibration split (full condition)",
                        "temperature": temperature},
        "decision_rule": "argmax cosine; Unknown argmax abstains; ties to the earlier candidate",
        "python_version": platform.python_version(), "torch_version": torch.__version__,
    }
    with open(os.path.join(out, "manifest.json"), "wb") as f:
        f.write(canonical_bytes(manifest) + b"\n")
    with open(os.path.join(out, "predictions.jsonl"), "w") as pf, open(os.path.join(out, "raw.jsonl"), "w") as rf:
        for record, payload, cs, sc in zip(records, payloads, cands, scores):
            probs = softmax(sc, temperature)
            best = max(range(len(cs)), key=sc.__getitem__)
            value = cs[best][1]
            pf.write(json.dumps({"id": record["id"], "status": "abstained" if value is None else "answered",
                                 "value": value, "probabilities": {c[0]: p for c, p in zip(cs, probs)},
                                 "confidence_source": "softmax_cosine_calibration_split_temperature"},
                                allow_nan=False) + "\n")
            rf.write(json.dumps({"id": record["id"], "payload_sha256": digest(payload), "condition": condition,
                                 "candidates": [c[2] for c in cs], "cosine": sc}, ensure_ascii=False) + "\n")
    completion = {"status": "complete", "completed_count": len(records),
                  "finished_at": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": file_digest(os.path.join(out, "manifest.json")),
                  "predictions_sha256": file_digest(os.path.join(out, "predictions.jsonl")),
                  "raw_sha256": file_digest(os.path.join(out, "raw.jsonl"))}
    with open(os.path.join(out, "completion.json"), "wb") as f:
        f.write(canonical_bytes(completion) + b"\n")


def summarize(report, records, scores, cands):
    """Leaderboard columns from the harness report plus abstention counts."""
    abstained = [max(range(len(s)), key=s.__getitem__) == len(s) - 1 for s in scores]
    tracks = {}
    for track, info in report["capability"]["tracks"].items():
        fams = info["families"].values()
        tracks[track] = f"{sum(f['correct'] for f in fams)}/{sum(f['total'] for f in fams)}"
    return {"n": len(records), "scored": True,
            "accuracy": report["cluster_accuracy_ci"]["estimate"],
            "accuracy_ci": [report["cluster_accuracy_ci"]["low"], report["cluster_accuracy_ci"]["high"]],
            "tracks": tracks,
            "correct_unknown": f"{sum(a for a, r in zip(abstained, records) if r['gold'] is None)}/"
                               f"{sum(r['gold'] is None for r in records)}",
            "false_abstention": f"{sum(a for a, r in zip(abstained, records) if r['gold'] is not None)}/"
                                f"{sum(r['gold'] is not None for r in records)}",
            "ece": report["probability_quality"]["ece"], "brier": report["probability_quality"]["brier"],
            "contrast_sets": report.get("contrast_sets", {}).get("accuracy"),
            "dataset_status": report["dataset_status"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="imajev-bench/ from scripts/download_data.py")
    ap.add_argument("--harness", required=True, help="imajev-harness/src from scripts/download_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    sys.path.insert(0, args.harness)

    root = args.dataset
    records_path = os.path.join(root, "records", "records-public.jsonl")
    records = [json.loads(line) for line in open(records_path, encoding="utf-8")]

    sys.path.insert(0, args.model)
    from modeling_metaencoder import MetaEncoder
    model = MetaEncoder.from_pretrained(args.model)

    conditions = args.conditions.split(",")
    encoded = {}
    for condition in conditions:
        for split in SPLITS:
            rs = [r for r in records if r["split"] == split]
            t0 = time.time()
            encoded[(condition, split)] = (rs, *encode_split(model, rs, root, condition, args.batch_size))
            print(f"[imajev] encoded {split}/{condition}: {len(rs)} items in {time.time() - t0:.0f}s", flush=True)

    rs, _, cands, scores = encoded[("full", "calibration")]
    temperature = fit_temperature([(s, [c[1] for c in cs].index(r["gold"]) if r["gold"] is not None else len(cs) - 1)
                                   for r, cs, s in zip(rs, cands, scores)])
    print(f"[imajev] calibration-split temperature: {temperature}", flush=True)

    summary = {"suite": "imajev", "metric": "accuracy", "temperature": temperature, "runs": {}}
    for (condition, split), (rs, payloads, cands, scores) in encoded.items():
        run_dir = os.path.join(args.out, f"{split}-{condition}")
        write_run(run_dir, rs, payloads, cands, scores, temperature, condition, args.model)
        if split == "test":
            summary["runs"][f"{split}-{condition}"] = {"n": len(rs), "scored": False,
                                                        "note": "labels withheld; submit this run folder"}
            continue
        score_path = os.path.join(args.out, f"{split}-{condition}.score.json")
        subprocess.run([sys.executable, "-m", "imajev_bench", "score", "--records", records_path, "--root", root,
                        "--allow-draft", "--split", split, "--predictions", os.path.join(run_dir, "predictions.jsonl"),
                        "--output", score_path], check=True, env={**os.environ, "PYTHONPATH": args.harness})
        summary["runs"][f"{split}-{condition}"] = summarize(json.load(open(score_path)), rs, scores, cands)
        print(f"[imajev] {split}/{condition}: {summary['runs'][f'{split}-{condition}']}", flush=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
