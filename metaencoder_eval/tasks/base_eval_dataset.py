# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from functools import wraps
from datasets import Dataset


# Schema for evaluation dataset, not used in the code.

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


class ImageVideoInstance:
    """
    len(bytes) == len(path) == len(resolution) == 1: image
    len(bytes) == len(path) == len(resolution) > 1: multi-image / video
    """
    def __init__(self, bytes, paths, resolutions):
        assert len(bytes) == len(paths) == len(resolutions)
        self.bytes = bytes
        self.paths = paths
        self.resolutions = resolutions

    def to_dict(self):
        return {
            "bytes": self.bytes,
            "paths": self.paths,
            "resolutions": self.resolutions,
        }


class AutoEvalPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoEvalPairDataset.registry:
            AutoEvalPairDataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs)
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                # print(f"Adding {dataset_name}")
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass


def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data.get('query_text', batch_data.get('cand_text', [])))
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        return batch_data

    return wrapper


def generate_cand_dataset(dataset, corpus):
    """
    Used for generating candidate datasets.
    Flatten candidates, merge with corpus, deduplication.
    Carries an optional per-candidate `cand_audio` (for audio-corpus tasks like Clotho);
    when absent the candidates are text/image-only and `cand_audio` is None.
    """
    has_audio = ("cand_audio" in dataset.column_names) or (corpus is not None and "cand_audio" in corpus.column_names)
    # Global-eval query datasets may omit per-row candidates entirely (candidates come only
    # from the corpus); only flatten per-row candidates when the columns exist.
    has_row_cands = "cand_text" in dataset.column_names and "cand_image" in dataset.column_names
    cand_rows = []
    all_cand_name = set()
    for row in (dataset if has_row_cands else []):
        if not row["cand_text"]:
            continue  # global-eval query row with no per-row candidates (corpus supplies them)
        cand_audio = row.get("cand_audio") if has_audio else None
        row_cand_names = row["dataset_infos"].get("cand_names", [])
        assert len(row["cand_text"]) == len(row["cand_image"]) == len(row_cand_names)
        for i, (cand_text, cand_image, cand_name) in enumerate(
                zip(row["cand_text"], row["cand_image"], row_cand_names)):
            if cand_name not in all_cand_name:
                r = {"cand_text": [cand_text], "cand_image": [cand_image],
                     "dataset_infos": {"cand_name": cand_name}}
                if has_audio:
                    r["cand_audio"] = [cand_audio[i] if cand_audio is not None else None]
                cand_rows.append(r)
                all_cand_name.add(cand_name)

    if corpus is not None:
        corpus_has_audio = "cand_audio" in corpus.column_names
        for row in corpus:
            assert len(row["cand_text"]) == len(row["cand_image"]) == len(row["dataset_infos"]["cand_names"]) == 1
            cand_name = row["dataset_infos"]["cand_names"][0]
            if cand_name not in all_cand_name:
                r = {"cand_text": row["cand_text"], "cand_image": row["cand_image"],
                     "dataset_infos": {"cand_name": cand_name}}
                if has_audio:
                    ca = row.get("cand_audio") if corpus_has_audio else None
                    r["cand_audio"] = [ca[0] if ca is not None else None]
                cand_rows.append(r)
                all_cand_name.add(cand_name)

    cand_dataset = Dataset.from_list(cand_rows)
    return cand_dataset
