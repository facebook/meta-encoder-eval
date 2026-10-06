#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=metaencoder-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
# Usage: MODEL=/path/to/meta-encoder DATA=/path/to/data OUT=results sbatch scripts/eval_all.sbatch
set -euo pipefail
: "${MODEL:?}" "${DATA:?}" "${OUT:?}"
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 run_eval.py --model "$MODEL" --data-root "$DATA" --out "$OUT"
python -m metaencoder_eval.jevbench --model "$MODEL" --data-dir "$DATA/jevbench" --out "$OUT/jevbench"
python -m metaencoder_eval.imajev --model "$MODEL" --dataset "$DATA/imajev-bench" \
    --harness "$DATA/imajev-harness/src" --out "$OUT/imajev"
python compare.py "$OUT" --per-task
