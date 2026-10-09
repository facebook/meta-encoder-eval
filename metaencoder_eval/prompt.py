# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""Prompt-building constants shared by the task parsers.

This is the Muse Glimmer subset of the training repo's `src/model/processor.py`. The parsers emit
query/candidate strings in that repo's wire format, and `encoder.to_item` turns those strings back
into `MetaEncoder` items:

    [<|patch|> ...][<|video|> ]{instruction}\ue000{content}

`\ue000` separates the instruction from the content. Strings without it (bare labels) are pure
content.
"""

MUSE_GLIMMER = "muse_glimmer"

INSTRUCTION_SEP = "\ue000"
IMAGE_PLACEHOLDER_TOKEN = "<|image_1|>"

VLM_IMAGE_TOKENS = {MUSE_GLIMMER: "<|patch|>"}
VLM_VIDEO_TOKENS = {MUSE_GLIMMER: "<|video|>"}


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    prompt = (instruction or "").strip() + INSTRUCTION_SEP + (text or "")
    if add_video_token:
        prompt = VLM_VIDEO_TOKENS[MUSE_GLIMMER] + " " + prompt
    if add_image_token:
        prompt = VLM_IMAGE_TOKENS[MUSE_GLIMMER] + " " + prompt
    return prompt


def resolve_candidate_instruction(instruction, model_backbone):
    return instruction
