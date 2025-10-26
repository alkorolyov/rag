from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from numpy.typing import NDArray
import numpy as np

from rag.storage.models import Document, SearchResult


class DocumentNotFoundError(KeyError):
    """Raised when a document is not found in the store."""
    pass


class ChunkNotFoundError(KeyError):
    """Raised when a chunk is not found in the store."""
    pass


class BaseDocumentStore(ABC):
    """
    Abstract base class for document storage.

    DocumentStore is responsible for storing and retrieving document chunks
    with their full text content and meta. It is the source of truth for
    document data, separate from vector embeddings.
    """

    # ===== Write Operations =====
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add parent documents to the store.

        Args:
            documents: List of parent Document objects (doc_type="parent")

        Returns:
            List of document IDs that were added

        Raises:
            ValueError: If documents are invalid or not parent type
        """
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[Document]) -> List[str]:
        """
        Add document chunks to the store.

        Args:
            chunks: List of chunk Document objects (doc_type="chunk")

        Returns:
            List of chunk IDs that were added

        Raises:
            ValueError: If chunks are invalid or not chunk type
        """
        pass

    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete chunks by IDs.

        Args:
            chunk_ids: List of chunk identifiers to delete

        Raises:
            ChunkNotFoundError: If any chunk_id does not exist
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """
        Delete parent document and all its chunks.

        Args:
            doc_id: Parent document identifier

        Raises:
            DocumentNotFoundError: If document does not exist
        """
        pass

    # ===== Read Operations - Chunks =====
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Document:
        """
        Get single chunk by ID.

        Args:
            chunk_id: The chunk identifier

        Returns:
            The Document object

        Raises:
            ChunkNotFoundError: If chunk_id does not exist
        """
        pass

    @abstractmethod
    def get_chunks(self, chunk_ids: List[str]) -> List[Document]:
        """
        Get multiple chunks by IDs (preserves order).

        This method is lenient and skips missing chunk IDs without raising
        an exception. Use this for batch retrieval after vector search.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            List of found Document objects in same order as input
        """
        pass

    @abstractmethod
    def get_document_chunks(self, doc_id: str) -> List[Document]:
        """
        Get all chunks belonging to a parent document.

        Args:
            doc_id: The parent document identifier

        Returns:
            List of Document chunks sorted by chunk index
        """
        pass

    @abstractmethod
    def filter_chunks(self, filters: Dict[str, Any]) -> List[Document]:
        """
        Filter chunks by metadata fields.

        Args:
            filters: Dictionary of meta field filters
                    Example: {"year": 2023, "source": "pubmed"}

        Returns:
            List of matching Documents (empty list if none match)
        """
        pass

    @abstractmethod
    def iter_chunks(self, batch_size: int = 100):
        """
        Iterate over all chunks in batches (generator).

        Args:
            batch_size: Number of chunks to yield per batch

        Yields:
            Document objects in batches
        """
        pass

    @abstractmethod
    def get_chunks_paginated(self, offset: int = 0, limit: int = 100) -> List[Document]:
        """
        Get chunks with pagination (single DB query).

        Args:
            offset: Number of chunks to skip
            limit: Maximum number of chunks to return

        Returns:
            List of Document objects
        """
        pass

    # ===== Read Operations - Documents =====
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get parent document by ID.

        Args:
            doc_id: The parent document identifier

        Returns:
            The parent Document object, or None if not found
        """
        pass

    @abstractmethod
    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """
        Get multiple parent documents by IDs.

        Args:
            doc_ids: List of parent document identifiers

        Returns:
            List of found parent Documents (skips missing)
        """
        pass

    # ===== Stats =====
    @abstractmethod
    def count_chunks(self) -> int:
        """
        Get total number of chunks in the store.

        Returns:
            Number of chunks
        """
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """
        Get total number of parent documents in the store.

        Returns:
            Number of parent documents
        """
        pass


class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage.

    VectorStore is responsible for storing and searching vector embeddings.
    It only stores chunk IDs and their embeddings, not the full document text.
    Document text is retrieved separately from DocumentStore.
    """

    @abstractmethod
    def add(self, embeddings: NDArray[np.float32], chunk_ids: List[str]) -> None:
        """
        Add embeddings to the vector store.

        Args:
            chunk_ids: List of chunk identifiers
            embeddings: Numpy array of shape (n, dimension) with embedding vectors

        Raises:
            ValueError: If chunk_ids and embeddings lengths don't match
            ValueError: If embeddings have wrong dimension
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return
            filters: Optional meta filters (if supported by implementation)

        Returns:
            List of SearchResult objects sorted by score (highest first)
        """
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """
        Delete vectors by chunk IDs.

        Args:
            chunk_ids: List of chunk identifiers to delete

        Raises:
            ChunkNotFoundError: If any chunk_id does not exist
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get the total number of vectors in the store.

        Returns:
            Number of vectors
        """
        pass
