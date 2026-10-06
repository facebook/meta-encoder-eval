# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import os


from metaencoder_eval.tasks.hf_paths import EVAL_DATASET_HF_PATH
from metaencoder_eval.tasks.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from metaencoder_eval.utils.dataset_utils import load_hf_dataset, sample_dataset
from metaencoder_eval.utils.vision_utils import process_video_frames, load_frames
from metaencoder_eval.prompt import process_input_text
import cv2


TASK_PROMPT = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
OPTIONS = ['yes', 'no']
@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']
    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    batch_size = len(batch_dict['question']) if batch_dict['question'] else 0
    for video_name, query, answer, question_id in \
            zip(batch_dict['video_name'], batch_dict['question'], batch_dict['answer'], batch_dict['question_id']):
        query = process_input_text(TASK_PROMPT, model_backbone, text=query + '? (A) yes; (B) no.', add_video_token=True)
        query_texts.append([query])
        video_path = f'{video_root}/v_{video_name}.mp4'
        frame_dir = f'{frame_root}/v_{video_name}'
        frames = load_frames(frame_dir)
        if not frames:
            # print(f'Extracting frames for: {video_path}')
            os.makedirs(frame_dir, exist_ok=True)
            assert os.path.exists(video_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames_saved)
            frame_idx = 0
            saved_frames = 0
            while saved_frames < max_frames_saved:
                assert cap.isOpened(), "not cap.isOpened()"
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to specific frame
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frame_dir, f"{saved_frames:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                saved_frames += 1
                frame_idx += step
            cap.release()
            # print(f'[{DATASET_PARSER_NAME}] Extracted #frames: {saved_frames}, dumped to {frame_dir}')

        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        # print(f'[{DATASET_PARSER_NAME}] Loaded #frames: {len(qry_frame_paths)}, from {frame_dir}')
        qry_frames = {"bytes": [None] * len(qry_frame_paths), "paths": qry_frame_paths, "resolutions": [None] * len(qry_frame_paths)}
        query_images.append([qry_frames])
        cand_texts.append(OPTIONS)
        cand_images.append([None] * len(OPTIONS))
        dataset_info = {
            "question_id": question_id,
            "video_id": video_name,
            "query": query,
            "cand_names": OPTIONS,
            "answer": answer,
            "label_name": answer,
            "answer_idx": OPTIONS.index(answer),
            "qry_frame_paths": qry_frame_paths,
        }
        dataset_infos.append(dataset_info)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"dataset.map(): global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}



DATASET_PARSER_NAME = "activitynetqa"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_activitynetqa_dataset(model_args, data_args, *args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['video_export_dir'] = kwargs.get("video_export_dir", None)
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    return dataset, None
