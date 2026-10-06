# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""NanoBEIR (zeta-alpha-ai/Nano*): BEIR-style queries, corpus and qrels parquet files.

Queries rank against the task's whole corpus. Seven subsets whose BEIR counterpart was in
training use that task's e5-mistral query instruction; the other six use none.
"""
from datasets import Dataset
import pyarrow.parquet as pq

from metaencoder_eval.prompt import MUSE_GLIMMER, process_input_text
from metaencoder_eval.tasks.base_eval_dataset import AutoEvalPairDataset

DATASET_PARSER_NAME = "nanobeir"

# e5-mistral query instructions (without the 'Instruct: ... Query: ' wrapper), by subset.
INSTRUCTIONS = {
    "nanomsmarco": "Given a web search query, retrieve relevant passages that answer the query",
    "nanohotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "nanofever": "Given a claim, retrieve documents that support or refute the claim",
    "nanonfcorpus": "Given a question, retrieve relevant documents that best answer the question",
    "nanofiqa2018": "Given a financial question, retrieve user replies that best answer the question",
    "nanonq": "Given a question, retrieve passages that answer the question",
    "nanoquoraretrieval": "Given a question, retrieve questions that are semantically equivalent",
}
NO_MEDIA = {"bytes": [None], "paths": [None], "resolutions": [None]}


def _rows(path):
    return pq.read_table(path).to_pylist()


def _qrels(rows):
    qrels = {}
    for row in rows:
        qid = row.get("query_id") or row.get("query-id")
        did = row.get("corpus_id") or row.get("corpus-id")
        if qid is None or did is None:
            continue
        score = float(row.get("label", row.get("score", 1)))
        if score > 0:
            qrels.setdefault(str(qid), {})[str(did)] = score
    return qrels


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_nanobeir_dataset(model_args, data_args, *args, **kwargs):
    subset = kwargs["subset_name"]
    instruction = INSTRUCTIONS.get(subset.lower().rsplit("/", 1)[-1], "")
    qrels = _qrels(_rows(kwargs["qrels_file"]))
    global_name = f"{kwargs.get('dataset_name', DATASET_PARSER_NAME)}/{subset}"

    queries = []
    for row in _rows(kwargs["query_file"]):
        qid, text = row.get("_id") or row.get("id"), row.get("text") or ""
        if qid is None or not (text or instruction):
            continue
        rel = qrels.get(str(qid), {})
        queries.append({
            "query_text": [process_input_text(instruction, MUSE_GLIMMER, text=text)],
            "query_image": [NO_MEDIA],
            "cand_text": [],
            "cand_image": [],
            "dataset_infos": {"qry_id": str(qid), "label_name": list(rel), "cand_names": [],
                              "rel_scores": list(rel.values()) or None},
            "global_dataset_name": global_name,
        })
    if queries and not any(q["dataset_infos"]["label_name"] for q in queries):
        raise ValueError(f"No positive labels loaded for {subset}; check the qrels file.")

    corpus = []
    for row in _rows(kwargs["candidate_file"]):
        did = row.get("_id") or row.get("id")
        title, text = row.get("title") or "", row.get("text") or ""
        text = f"{title}\n{text}" if title and text else text
        if did is None or not text:
            continue
        corpus.append({"cand_text": [text], "cand_image": [NO_MEDIA], "dataset_infos": {"cand_names": [str(did)]}})

    print(f"{subset}: {len(queries)} queries, {len(corpus)} documents")
    return Dataset.from_list(queries), Dataset.from_list(corpus)
