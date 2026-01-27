"""Hybrid retrieval combining dense and sparse search.

This module implements the CV claim:
"Built biomedical RAG workflows using Qdrant and OpenSearch (BM25)"
"""

from dataclasses import dataclass
from typing import Callable

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cuda"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "bioasq"

    # Retrieval
    dense_k: int = 50
    sparse_k: int = 50
    dense_weight: float = 0.5

    @property
    def sparse_weight(self) -> float:
        return 1.0 - self.dense_weight

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


class HybridRetriever:
    """Hybrid retriever combining Qdrant (dense) and BM25 (sparse) with RRF fusion.

    Example:
        >>> config = HybridConfig(collection_name="my-collection")
        >>> retriever = HybridRetriever(config)
        >>> retriever.index(documents)
        >>> results = retriever.search("What causes diabetes?", k=10)
    """

    def __init__(self, config: HybridConfig):
        self.config = config
        self._embeddings = None
        self._qdrant_client = None
        self._vectorstore = None
        self._bm25 = None
        self._ensemble = None

    @property
    def embeddings(self) -> HuggingFaceBgeEmbeddings:
        """Lazy-load embedding model."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceBgeEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.config.embedding_device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def qdrant_client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
            )
        return self._qdrant_client

    def index(self, documents: list[Document], force_recreate: bool = False) -> None:
        """Index documents for both dense and sparse retrieval.

        Args:
            documents: List of LangChain Documents
            force_recreate: If True, recreate Qdrant collection
        """
        # Dense index (Qdrant)
        collection_exists = self.config.collection_name in [
            c.name for c in self.qdrant_client.get_collections().collections
        ]

        if collection_exists and not force_recreate:
            self._vectorstore = Qdrant(
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                embeddings=self.embeddings,
            )
        else:
            self._vectorstore = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=self.config.qdrant_url,
                collection_name=self.config.collection_name,
                force_recreate=force_recreate,
            )

        # Sparse index (BM25)
        self._bm25 = BM25Retriever.from_documents(documents)
        self._bm25.k = self.config.sparse_k

        # Ensemble
        dense_retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.config.dense_k}
        )

        self._ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25],
            weights=[self.config.dense_weight, self.config.sparse_weight],
        )

    def search(self, query: str, k: int = 10) -> list[Document]:
        """Search using hybrid retrieval.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of Documents ranked by RRF fusion score
        """
        if self._ensemble is None:
            raise ValueError("Call index() before search()")

        results = self._ensemble.invoke(query)

        # Deduplicate by doc_id
        seen = set()
        unique = []
        for doc in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
            if len(unique) >= k:
                break

        return unique

    def search_dense(self, query: str, k: int = 10) -> list[Document]:
        """Search using only dense (Qdrant) retrieval."""
        if self._vectorstore is None:
            raise ValueError("Call index() before search()")
        return self._vectorstore.similarity_search(query, k=k)

    def search_sparse(self, query: str, k: int = 10) -> list[Document]:
        """Search using only sparse (BM25) retrieval."""
        if self._bm25 is None:
            raise ValueError("Call index() before search()")
        return self._bm25.invoke(query)[:k]

    def search_with_filter(
        self,
        query: str,
        k: int = 10,
        entity_filter: dict[str, str | list[str]] | None = None,
    ) -> list[Document]:
        """Search with entity-based filtering.

        Args:
            query: Search query
            k: Number of results
            entity_filter: Filter by entity type and value(s).
                Example: {"entity_GENE": "BRCA1"}
                Example: {"entity_DISEASE": ["cancer", "tumor"]}

        Returns:
            Filtered documents ranked by similarity
        """
        if self._vectorstore is None:
            raise ValueError("Call index() before search()")

        if not entity_filter:
            return self.search(query, k)

        # Build Qdrant filter
        conditions = []
        for field, value in entity_filter.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=field, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )

        qdrant_filter = Filter(must=conditions)

        # Search with filter (dense only - BM25 doesn't support filtering)
        results = self._vectorstore.similarity_search(
            query, k=k, filter=qdrant_filter
        )

        return results
