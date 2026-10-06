# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""Download the evaluation data into one directory.

    python scripts/download_data.py --data-root /path/to/data [--only mmeb,videommmu,nanobeir,jevbench]

Layout produced (paths the suite YAMLs expect, relative to --data-root):

    vlm2vec_eval/        TIGER-Lab/MMEB-V2 image/video frames and visdoc images
    MMEB-V3/image_tasks/MCMR/
    vlm2vec_eval/video-tasks/videos/video_qa/VideoMMMU/{Perception,Comprehension,Adaptation}/
    nanobeir/Nano*/      zeta-alpha-ai NanoBEIR
    jevbench/            JEVBench public split
    imajev-bench/        ImajevBench v2.0-lite records and images
    imajev-harness/      ImajevBench harness (github.com/mohit67890/imajev), which scores runs

The MMEB test files (queries/candidates/qrels) are fetched from the Hub at evaluation time.
MMLU and MMMU are built by scripts/prepare_closed_set.py. MMEB-V2 is several hundred GB.
"""
import argparse
import glob
import os
import subprocess
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download

NANOBEIR = ["NanoArguAna", "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", "NanoFiQA2018", "NanoHotpotQA",
            "NanoMSMARCO", "NanoNFCorpus", "NanoNQ", "NanoQuoraRetrieval", "NanoSCIDOCS", "NanoSciFact",
            "NanoTouche2020"]
VIDEOMMMU_ZIPS = ["Art.zip", "Business.zip", "Engineering.zip", "Humanities.zip", "Medicine.zip", "Science.zip"]
JEVBENCH_URL = "https://raw.githubusercontent.com/fstandhartinger/jevbench/bb05a335bc809e61b20c0f745d25499a82b326fc/datasets/public/{}.jsonl"
MMEB_V2_REVISION = "e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1"
MMEB_V3_REVISION = "4a5560b2b64384204b6fea8a82ea986eba51f5aa"
VIDEOMMMU_REVISION = "d1c35ac933123d79e877b7f1b9506afb0309cf1b"
IMAJEV_REVISION = "043ab1f7425ef7b60e6206199cff43e1f80bc64f"
IMAJEV_HARNESS_URL = "https://github.com/mohit67890/imajev/archive/ccf586d43d2a580319b6535c893668904d909eb9.tar.gz"


def untar(archives, dest, strip=0):
    os.makedirs(dest, exist_ok=True)
    cat = subprocess.Popen(["cat", *archives], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xzf", "-", "-C", dest, f"--strip-components={strip}"], stdin=cat.stdout, check=True)
    cat.wait()


def mmeb(root):
    src = snapshot_download("TIGER-Lab/MMEB-V2", repo_type="dataset", revision=MMEB_V2_REVISION,
                            local_dir=os.path.join(root, "_MMEB-V2"),
                            allow_patterns=["image-tasks/*", "video-tasks/frames/*", "visdoc-tasks/visdoc-tasks.images.tar.gz"])
    out = os.path.join(root, "vlm2vec_eval")
    frames = os.path.join(out, "video-tasks", "frames")
    # Archive roots: MMEB/<task>, images/<task>, <task> (cls), data/ziyan/... (ret),
    # video_qa/<task>, video_mret/<task>.
    untar([os.path.join(src, "image-tasks", "mmeb_v1.tar.gz")], os.path.join(out, "image-tasks"), strip=1)
    untar([os.path.join(src, "visdoc-tasks", "visdoc-tasks.images.tar.gz")], os.path.join(out, "visdoc-tasks"))
    for task, dest in [("cls", "video_cls"), ("ret", "video_ret"), ("qa", ""), ("mret", "")]:
        parts = sorted(glob.glob(os.path.join(src, "video-tasks", "frames", f"video_{task}.tar.gz*")))
        untar(parts, os.path.join(frames, dest))

    mcmr = hf_hub_download("VLM2Vec/MMEB-V3", "image_tasks/MCMR.tar.gz", repo_type="dataset", revision=MMEB_V3_REVISION,
                           local_dir=os.path.join(root, "_MMEB-V3"))
    mcmr_dir = os.path.join(root, "MMEB-V3", "image_tasks")
    untar([mcmr], mcmr_dir)
    inner = os.path.join(mcmr_dir, "MCMR", "images.tar.gz")
    if os.path.exists(inner) and not os.path.isdir(os.path.join(mcmr_dir, "MCMR", "images")):
        untar([inner], os.path.join(mcmr_dir, "MCMR"))


def videommmu(root):
    # Gated on the Hub: accept the terms at https://huggingface.co/datasets/lmms-eval/VideoMMMU first.
    base = os.path.join(root, "vlm2vec_eval", "video-tasks", "videos", "video_qa", "VideoMMMU")
    pool = os.path.join(base, "Perception")
    os.makedirs(pool, exist_ok=True)
    for name in VIDEOMMMU_ZIPS:
        path = hf_hub_download("lmms-eval/VideoMMMU", name, repo_type="dataset", revision=VIDEOMMMU_REVISION,
                               local_dir=os.path.join(base, "_raw"))
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if member.endswith(".mp4"):
                    target = os.path.join(pool, os.path.basename(member))
                    if not os.path.exists(target):
                        with z.open(member) as s, open(target, "wb") as d:
                            d.write(s.read())
    # Every subset reuses the same clips, keyed by question id.
    for subset in ("Comprehension", "Adaptation"):
        link = os.path.join(base, subset)
        if not os.path.exists(link):
            os.symlink(pool, link)


def nanobeir(root):
    for name in NANOBEIR:
        snapshot_download(f"zeta-alpha-ai/{name}", repo_type="dataset",
                          local_dir=os.path.join(root, "nanobeir", name))


def jevbench(root):
    out = os.path.join(root, "jevbench")
    os.makedirs(out, exist_ok=True)
    for split in ("original", "easy", "hard"):
        urllib.request.urlretrieve(JEVBENCH_URL.format(split), os.path.join(out, f"{split}.jsonl"))


def imajev(root):
    # Real files, not cache symlinks: the harness rejects image paths that resolve outside the dataset.
    snapshot_download("mohit67890/imajev-bench", repo_type="dataset", revision=IMAJEV_REVISION,
                      local_dir=os.path.join(root, "imajev-bench"), allow_patterns=["records/*", "assets/*"])
    archive = os.path.join(root, "imajev-harness.tar.gz")
    urllib.request.urlretrieve(IMAJEV_HARNESS_URL, archive)
    untar([archive], os.path.join(root, "imajev-harness"), strip=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--only", default="mmeb,videommmu,nanobeir,jevbench,imajev")
    args = ap.parse_args()
    steps = {"mmeb": mmeb, "videommmu": videommmu, "nanobeir": nanobeir, "jevbench": jevbench, "imajev": imajev}
    for name in args.only.split(","):
        print(f"== {name}", flush=True)
        steps[name](args.data_root)


if __name__ == "__main__":
    main()
