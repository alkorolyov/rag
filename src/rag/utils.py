import itertools
from typing import Callable

import numpy as np
import tiktoken
from transformers import AutoTokenizer


def batched(iterable, size=100):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def embed_biorag_datasets(ds, test_ds, embedder):
    def embed(batch, column):
        embs = embedder.embed_batch(batch[column])
        return {'embedding': embs}

    ds = ds.map(
        embed,
        batch_size=8,
        batched=True,
        fn_kwargs={'column': 'passage'},
    )
    test_ds = test_ds.map(
        embed,
        batch_size=8,
        batched=True,
        fn_kwargs={'column': 'question'},
    )
    return ds, test_ds


def precision_at_k(hit_flags):
    return float(hit_flags.mean(axis=1).mean())


def recall_at_k(hit_flags, qrels_counts):
    retrieved = hit_flags.sum(axis=1)
    recall = retrieved / qrels_counts
    return float(recall.mean())


def mrr_at_k(hit_flags):
    has_hit = hit_flags.any(axis=1)
    ranks = np.argmax(hit_flags, axis=1) + 1
    reciprocal = np.zeros_like(ranks, dtype=np.float32)
    reciprocal[has_hit] = 1.0 / ranks[has_hit]
    return float(reciprocal.mean())


def ndcg_at_k(hit_flags: np.ndarray, qrels_counts) -> float:
    # Binary relevance (1 if retrieved id is in gold set)
    # DCG = sum_{i=1..k} (rel_i / log2(i+1)), since rel∈{0,1} we can simplify
    k = hit_flags.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k+2))  # [1/log2(2), 1/log2(3), ...]
    dcg = (hit_flags * discounts).sum(axis=1)
    # Ideal DCG for binary relevance = sum of top min(k, |gold|) discounts
    # We need |gold| per query:
    ndcg = dcg / np.array([discounts[:min(k, c)].sum() if c > 0 else 1.0
                           for c in qrels_counts])
    return float(ndcg.mean())


def get_hit_flags(retrieved_ids, qrels):
    hit_flags = np.zeros_like(retrieved_ids, dtype=np.bool)
    for i in range(len(retrieved_ids)):
        hit_flags[i] = np.isin(retrieved_ids[i], qrels[i])
    return hit_flags


def get_metrics(retrieved_ids, query_ds, k):
    qrels = [np.array(eval(gold)) for gold in query_ds['relevant_passage_ids']]
    qrels_counts = [len(s) for s in qrels]

    hit_flags = get_hit_flags(retrieved_ids, qrels)

    return {
        f"P@{k}": precision_at_k(hit_flags),
        f"R@{k}": recall_at_k(hit_flags, qrels_counts),
        f"MRR@{k}": mrr_at_k(hit_flags),
        f"nDCG@{k}": ndcg_at_k(hit_flags, qrels_counts),
    }


def embed_dataset(dataset, embedder, column='passage', batch_size=8):
    def embed(batch, column):
        embs = embedder.embed_batch(batch[column])
        return {'embedding': embs}

    return dataset.map(
        embed,
        batch_size=batch_size,
        batched=True,
        fn_kwargs={'column': column},
    )

def get_token_counter(model_name: str) -> Callable[[str], int]:
    if "openai" in model_name.lower():
        encoding = tiktoken.get_encoding(model_name)
        return lambda text: len(encoding.encode(text))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return lambda text: len(tokenizer.encode(text))