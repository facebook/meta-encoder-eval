# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate MetaEncoder on one or more suites.

    torchrun --nproc_per_node=8 run_eval.py --model /path/to/meta-encoder \
        --data-root /path/to/data --out results/ --suites mmeb_image,mmlu

Each task writes `{task}_score.json` and `{task}_pred.jsonl` under `--out/<suite>/`, and each
suite writes `--out/<suite>/summary.json` with the headline number. Finished tasks are skipped
on rerun.
"""
import argparse
import datetime
import json
import os
import types

import numpy as np
import torch
import torch.distributed as dist
import yaml

import metaencoder_eval.tasks  # noqa: F401  registers the parsers
from metaencoder_eval.encoder import DistributedEncoder, flatten_rows
from metaencoder_eval.metrics import RankingMetrics
from metaencoder_eval.tasks.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset

SUITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metaencoder_eval", "suites")

# suite -> headline metric; the suite score is its unweighted mean over tasks
SUITES = {
    "mmeb_image": "hit@1",
    "mmeb_video": "hit@1",
    "mmeb_visdoc": "ndcg_linear@5",
    "videommmu": "accuracy",
    "nanobeir": "ndcg_linear@10",
    "mmlu": "hit@1",
    "mmmu": "hit@1",
}
PATH_KEYS = ["image_root", "video_root", "frame_root", "clip_root", "data_path",
             "query_file", "candidate_file", "qrels_file", "dataset_path"]
# eval.py ran these at batch size 4 to fit memory
SMALL_BATCH_TASKS = {"Charades-STA", "QVHighlight", "MomentSeeker", "YouCook2", "Video-MME"}
MODEL_ARGS = types.SimpleNamespace(model_backbone="muse_glimmer")
DATA_ARGS = types.SimpleNamespace(image_resolution=None)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def load_task(cfg):
    # Parsers can write frames/stills to disk on first use, so rank 0 builds the cache first.
    if is_main():
        qry, corpus = AutoEvalPairDataset.instantiate(model_args=MODEL_ARGS, data_args=DATA_ARGS, **cfg)
    barrier()
    if not is_main():
        qry, corpus = AutoEvalPairDataset.instantiate(model_args=MODEL_ARGS, data_args=DATA_ARGS, **cfg)
    return qry, generate_cand_dataset(qry, corpus)


def score(qry_emb, qry_infos, cand_emb, cand_names, eval_type):
    preds = []
    if eval_type == "global":
        ranked = np.argsort(-(qry_emb @ cand_emb.T), axis=1)
        for order, info in zip(ranked, qry_infos):
            labels = info["label_name"] if isinstance(info["label_name"], list) else [info["label_name"]]
            preds.append({"prediction": [cand_names[i] for i in order], "label": labels,
                          "rel_scores": info.get("rel_scores")})
    else:
        index = {name: i for i, name in enumerate(cand_names)}
        for q, info in zip(qry_emb, qry_infos):
            names = info["cand_names"]
            order = np.argsort(-(cand_emb[[index[n] for n in names]] @ q))
            labels = info["label_name"] if isinstance(info["label_name"], list) else [info["label_name"]]
            preds.append({"prediction": [names[i] for i in order], "label": labels,
                          "rel_scores": info.get("rel_scores")})
    return preds


def run_task(encoder, name, cfg, out_dir, batch_size):
    score_path = os.path.join(out_dir, f"{name}_score.json")
    if os.path.exists(score_path):
        return json.load(open(score_path))
    os.makedirs(os.path.dirname(score_path), exist_ok=True)

    eval_type = cfg.get("eval_type", "global")
    metrics = cfg.pop("metrics", None) or ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
    num_questions = cfg.pop("num_questions", None)
    qry_ds, cand_ds = load_task(cfg)
    qry_rows, cand_rows = list(qry_ds), list(cand_ds)

    encoder.batch_size = 4 if name in SMALL_BATCH_TASKS else batch_size
    tag = name.replace("/", "_")
    qry_emb = encoder.encode(*flatten_rows(qry_rows, "qry"), tag=f"{tag}.qry")
    cand_emb = encoder.encode(*flatten_rows(cand_rows, "cand"), tag=f"{tag}.cand")

    result = None
    if is_main():
        preds = score(qry_emb, [r["dataset_infos"] for r in qry_rows], cand_emb,
                      [r["dataset_infos"]["cand_name"] for r in cand_rows], eval_type)
        result = RankingMetrics(metrics).evaluate(preds)
        result["num_pred"] = result["num_data"] = len(preds)
        if num_questions:
            # Official accuracy: questions this task cannot score (open-ended, missing media)
            # stay in the denominator and count as wrong.
            result["num_questions"] = num_questions
            result["accuracy"] = result["hit@1"] * len(preds) / num_questions
        with open(os.path.join(out_dir, f"{name}_pred.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        json.dump(result, open(score_path, "w"), indent=2)
        print(f"[{name}] " + " ".join(f"{k}={result[k]:.4f}" for k in ("hit@1", "accuracy", "ndcg_linear@5", "ndcg_linear@10") if k in result), flush=True)
    barrier()
    return result


def run_suite(encoder, suite, args):
    metric = SUITES[suite]
    tasks = yaml.safe_load(open(os.path.join(SUITE_DIR, f"{suite}.yaml")))
    if args.tasks:
        keep = set(args.tasks.split(","))
        tasks = {k: v for k, v in tasks.items() if k in keep}
    out_dir = os.path.join(args.out, suite)
    scores = {}
    for name, cfg in tasks.items():
        cfg = dict(cfg)
        for key in PATH_KEYS:
            if cfg.get(key) and not os.path.isabs(cfg[key]):
                cfg[key] = os.path.join(args.data_root, cfg[key])
        res = run_task(encoder, name, cfg, out_dir, args.batch_size)
        if res is not None:
            scores[name] = res[metric]
    if is_main() and scores:
        summary = {"suite": suite, "metric": metric, "score": float(np.mean(list(scores.values()))),
                   "num_tasks": len(scores), "tasks": scores}
        json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
        print(f"== {suite}: {metric} = {summary['score']:.4f} over {len(scores)} tasks", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local snapshot of facebook/meta-encoder")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--tasks", default="", help="comma-separated subset of task names")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2),
                                device_id=torch.device(f"cuda:{local_rank}"))

    encoder = DistributedEncoder(args.model, batch_size=args.batch_size,
                                 gather_dir=os.path.join(args.out, ".gather"))
    for suite in args.suites.split(","):
        run_suite(encoder, suite.strip(), args)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
