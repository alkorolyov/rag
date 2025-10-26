import itertools
from typing import Callable, List, Dict, Any

import numpy as np
import pandas as pd
import tiktoken
from transformers import AutoTokenizer

from rag.embeddings import BaseEmbedder


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


def evaluate_retrieval(eval_data: List[Dict[str, Any]], k: int = 10) -> Dict[str, float]:
    """
    Calculate IR metrics from eval_data format (LlamaIndex-style).

    This function wraps the existing NumPy-optimized metric functions to work
    with the eval_data format commonly used in notebooks and RAGAS evaluation.

    Args:
        eval_data: List of dicts with keys:
            - 'retrieved_doc_ids': List[int] - retrieved document IDs (in rank order)
            - 'relevant_doc_ids': List[int] - ground truth relevant document IDs
        k: Cutoff for metrics calculation (default: 10)

    Returns:
        Dict with aggregated metrics:
            - P@k: Precision at k (what % of retrieved docs are relevant)
            - R@k: Recall at k (what % of relevant docs were retrieved)
            - MRR@k: Mean Reciprocal Rank (average 1/rank of first relevant doc)
            - nDCG@k: Normalized Discounted Cumulative Gain (quality of ranking)
            - Hit@k: Hit rate (% of queries with at least one relevant doc)

    Example:
        >>> eval_data = [
        ...     {'retrieved_doc_ids': [1, 2, 3], 'relevant_doc_ids': [1, 5]},
        ...     {'retrieved_doc_ids': [4, 5, 6], 'relevant_doc_ids': [5, 7]}
        ... ]
        >>> metrics = evaluate_retrieval(eval_data, k=10)
        >>> print(f"Precision@10: {metrics['P@10']:.4f}")
    """
    if not eval_data:
        raise ValueError("eval_data cannot be empty")

    # Step 1: Extract retrieved IDs and pad/truncate to k
    # We need a 2D numpy array of shape (n_queries, k)
    n_queries = len(eval_data)
    retrieved_ids = np.zeros((n_queries, k), dtype=np.int64)

    for i, item in enumerate(eval_data):
        ret_ids = item['retrieved_doc_ids'][:k]  # Take top-k
        retrieved_ids[i, :len(ret_ids)] = ret_ids
        # Remaining positions stay 0 (padding)

    # Step 2: Extract ground truth relevant IDs (qrels)
    qrels = [item['relevant_doc_ids'] for item in eval_data]
    qrels_counts = np.array([len(q) for q in qrels])

    # Step 3: Calculate hit flags using existing function
    hit_flags = get_hit_flags(retrieved_ids, qrels)

    # Step 4: Calculate metrics using existing optimized functions
    metrics = {
        f"P@{k}": precision_at_k(hit_flags),
        f"R@{k}": recall_at_k(hit_flags, qrels_counts),
        f"MRR@{k}": mrr_at_k(hit_flags),
        f"nDCG@{k}": ndcg_at_k(hit_flags, qrels_counts),
    }

    # Step 5: Add hit rate (not in original get_metrics)
    # Hit rate = % of queries where we retrieved at least 1 relevant doc
    hit_rate = float(hit_flags.any(axis=1).mean())
    metrics[f"Hit@{k}"] = hit_rate

    return metrics


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


def cosine_similarity_matrix(texts: List[str], embedder: BaseEmbedder) -> pd.DataFrame:
    """
    Create a cosine similarity matrix for a list of texts.

    Args:
        texts: List of strings to compare
        settings: Optional Settings object. If None, uses default settings.

    Returns:
        DataFrame with cosine similarities, indexed and columned by text labels

    Example:
        >>> from rag.config import settings
        >>> texts = ["heart attack", "myocardial infarction", "diabetes"]
        >>> sim_matrix = cosine_similarity_matrix(texts, settings)
        >>> print(sim_matrix)
    """
    from rag.embeddings import create_embedder
    from rag.config import settings as default_settings

    # Embed all texts
    embeddings = embedder.embed_batch(texts)

    # Calculate cosine similarity (embeddings are already normalized)
    # For normalized vectors: cosine_sim(a,b) = dot(a,b)
    similarity_matrix = np.dot(embeddings, embeddings.T)

    # Create DataFrame with text labels
    df = pd.DataFrame(
        similarity_matrix,
        index=texts,
        columns=texts
    )

    return df