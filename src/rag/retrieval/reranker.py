"""Reranking utilities for improving retrieval precision."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents by relevance to query."""
        ...


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cuda"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: List[Document], batch_size: int = 32) -> List[Document]:
        """Rerank documents using cross-encoder scores.

        Args:
            query: Query string
            documents: List of LangChain Documents
            batch_size: Batch size for inference

        Returns:
            Documents sorted by relevance (highest first), with scores in metadata
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Add scores to metadata
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        return sorted(documents, key=lambda d: d.metadata["rerank_score"], reverse=True)
