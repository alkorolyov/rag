"""Evaluation utilities for retrieval and RAG systems."""

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    precision: float
    recall: float
    mrr: float
    hit_rate: float
    k: int

    def __repr__(self) -> str:
        return (
            f"P@{self.k}={self.precision:.3f} | "
            f"R@{self.k}={self.recall:.3f} | "
            f"MRR@{self.k}={self.mrr:.3f} | "
            f"Hit@{self.k}={self.hit_rate:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            f"P@{self.k}": self.precision,
            f"R@{self.k}": self.recall,
            f"MRR@{self.k}": self.mrr,
            f"Hit@{self.k}": self.hit_rate,
        }


def evaluate_retriever(
    retriever,
    queries: list[str],
    qrels: list[set],
    k: int = 10,
    n_samples: int | None = None,
    seed: int = 42,
) -> RetrievalMetrics:
    """Evaluate retriever with standard IR metrics.

    Args:
        retriever: Object with invoke(query) -> List[Document] method
        queries: List of query strings
        qrels: List of relevant doc_id sets (one per query)
        k: Cutoff for metrics
        n_samples: Sample size for faster evaluation (None = all)
        seed: Random seed for sampling

    Returns:
        RetrievalMetrics with averaged scores
    """
    if n_samples and n_samples < len(queries):
        np.random.seed(seed)
        idx = np.random.choice(len(queries), n_samples, replace=False)
        queries = [queries[i] for i in idx]
        qrels = [qrels[i] for i in idx]

    precisions, recalls, mrrs, hits = [], [], [], []

    for query, relevant in tqdm(zip(queries, qrels), total=len(queries), desc="Evaluating"):
        results = retriever.invoke(query)
        retrieved = [d.metadata.get("doc_id") for d in results[:k]]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for doc_id in retrieved:
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc_id)
        retrieved = unique[:k]

        # Calculate metrics
        hit_flags = [1 if r in relevant else 0 for r in retrieved]
        n_rel = sum(hit_flags)

        precisions.append(n_rel / k if k > 0 else 0)
        recalls.append(n_rel / len(relevant) if relevant else 0)
        mrrs.append(next((1 / (i + 1) for i, h in enumerate(hit_flags) if h), 0))
        hits.append(1 if n_rel > 0 else 0)

    return RetrievalMetrics(
        precision=np.mean(precisions),
        recall=np.mean(recalls),
        mrr=np.mean(mrrs),
        hit_rate=np.mean(hits),
        k=k,
    )
