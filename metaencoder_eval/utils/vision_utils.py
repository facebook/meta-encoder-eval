# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import re
import shutil
import subprocess

import numpy as np
from PIL import Image

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def qa_template(question, candidates, answer):
    question = f"{question}\n"
    question += "Options:\n"
    answer_idx = -1
    options = []
    for idx, c in enumerate(candidates):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        options.append(f"({chr(ord('A') + idx)}) {c}")
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"
    return question, options, answer, answer_idx


def load_frames(frames_dir, filter_func=None):
    """Image files in `frames_dir`, in natural (numeric-aware) order."""
    def natural_sort_key(filename):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

    if not os.path.isdir(frames_dir):
        return []
    results = []
    for frame_name in sorted(os.listdir(frames_dir), key=natural_sort_key):
        if os.path.splitext(frame_name)[-1].lower() in IMAGE_EXTENSIONS:
            if filter_func is None or filter_func(frame_name):
                results.append(f"{frames_dir}/{frame_name}")
    return results


def sample_frames(frames, num_segments):
    frame_id_list = np.linspace(0, len(frames) - 1, num_segments, dtype=int).tolist()
    last_frame_id = frame_id_list[-1]
    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            sampled_frames.append(frames[frame_idx])
        except IndexError:
            break
    # Pad with the last frame when the clip has fewer frames than requested.
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames


def process_video_frames(frame_dir, num_frames=None):
    """Uniformly sample `num_frames` frame paths from `frame_dir` (all of them if None)."""
    if num_frames == 0:
        return []
    frames = load_frames(frame_dir)
    if num_frames is None:
        return frames
    elif num_frames and num_frames <= len(frames):
        frames = sample_frames(frames, num_segments=num_frames)
    return frames


def _have_ffmpeg():
    global _FFMPEG_OK
    try:
        return _FFMPEG_OK
    except NameError:
        pass
    _FFMPEG_OK = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    if not _FFMPEG_OK:
        print("[vision_utils] ffmpeg/ffprobe not on PATH; decoding frames with decord/cv2 instead")
    return _FFMPEG_OK


def _save_frames_pyav(video_path, frame_dir, max_frames_saved):
    """Frame extraction without the ffmpeg CLI, via decord or OpenCV."""
    idxs, frames = None, None
    try:
        import decord
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        if total == 0:
            return 0
        idxs = (list(range(total)) if total <= max_frames_saved
                else [math.floor(i * total / max_frames_saved) for i in range(max_frames_saved)])
        frames = vr.get_batch(idxs).asnumpy()
    except Exception:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return 0
        idxs = (list(range(total)) if total <= max_frames_saved
                else [math.floor(i * total / max_frames_saved) for i in range(max_frames_saved)])
        got = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                got.append(fr[:, :, ::-1])       # BGR -> RGB
        cap.release()
        frames = got
    if frames is None or len(frames) == 0:
        return 0
    for j, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(frame_dir, "frame_%04d.jpg" % (j + 1)), quality=90)
    return len(frames)


def save_frames(video_path, frame_dir, max_frames_saved, file_name_prefix=''):
    """Extract up to `max_frames_saved` uniformly spaced frames, unless `frame_dir` already has some."""
    if os.path.exists(frame_dir) and any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(frame_dir)):
        return
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File {video_path} does not exist")
    os.makedirs(frame_dir, exist_ok=True)
    if not _have_ffmpeg():
        if _save_frames_pyav(video_path, frame_dir, max_frames_saved) == 0:
            raise RuntimeError("could not decode any frame from %s" % video_path)
        return
    total_frames = get_total_frames(video_path)
    if total_frames == 0:
        print("No frames found in video.")
        return
    if total_frames <= max_frames_saved:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / max_frames_saved
        frame_indices = [math.floor(i * step) for i in range(max_frames_saved)]
    select_expr = "+".join([f"eq(n\\,{i})" for i in frame_indices])
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-vf", f"select='{select_expr}'",
           "-vsync", "vfr", os.path.join(frame_dir, "frame_%04d.jpg")]
    subprocess.run(cmd, check=True)


def get_total_frames(video_path):
    """Frame count from ffprobe (decoded count, falling back to the container's nb_frames)."""
    def probe(entry, extra=()):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", *extra,
               "-show_entries", f"stream={entry}", "-of", "default=nokey=1:noprint_wrappers=1", video_path]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    output = probe("nb_read_frames", ("-count_frames",))
    if output.isdigit():
        return int(output)
    fallback = probe("nb_frames")
    if fallback.isdigit():
        return int(fallback)
    raise ValueError(f"Could not determine total frames for {video_path}. Outputs: '{output}', fallback: '{fallback}'")
