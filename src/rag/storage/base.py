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
    with their full text content and metadata. It is the source of truth for
    document data, separate from vector embeddings.
    """

    @abstractmethod
    def add(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of chunk IDs that were added

        Raises:
            ValueError: If documents are invalid
        """
        pass

    @abstractmethod
    def get_by_id(self, chunk_id: str) -> Document:
        """
        Get a single document by chunk ID.

        Args:
            chunk_id: The chunk identifier

        Returns:
            The Document object

        Raises:
            ChunkNotFoundError: If chunk_id does not exist
        """
        pass

    @abstractmethod
    def get_by_ids(self, chunk_ids: List[str]) -> List[Document]:
        """
        Get multiple documents by chunk IDs.

        This method is lenient and skips missing chunk IDs without raising
        an exception. Use this for batch retrieval after vector search.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            List of found Document objects (may be shorter than input if some missing)
        """
        pass

    @abstractmethod
    def get_by_doc_id(self, doc_id: str | int) -> List[Document]:
        """
        Get all chunks belonging to a parent document.

        Args:
            doc_id: The parent document identifier

        Returns:
            List of Document chunks (empty list if none found)
        """
        pass

    @abstractmethod
    def get_parent_document(self, doc_id: str | int) -> Optional[Document]:
        """
        Get the parent document (full document, not a chunk).

        Args:
            doc_id: The parent document identifier (without '#' symbol)

        Returns:
            The parent Document object, or None if not found

        Note:
            This retrieves documents with doc_type="parent" in metadata
            or documents whose ID doesn't contain '#' symbol.
        """
        pass

    @abstractmethod
    def filter_by_metadata(self, filters: Dict[str, Any]) -> List[Document]:
        """
        Filter documents by metadata fields.

        Args:
            filters: Dictionary of metadata field filters
                    Example: {"year": 2023, "source": "pubmed"}

        Returns:
            List of matching Documents (empty list if none match)
        """
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """
        Delete documents by chunk IDs.

        Args:
            chunk_ids: List of chunk identifiers to delete

        Raises:
            ChunkNotFoundError: If any chunk_id does not exist
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get the total number of documents in the store.

        Returns:
            Number of documents
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterate over all documents in the store.

        Yields:
            Document objects

        Example:
            for doc in doc_store:
                print(doc.id, doc.text[:50])
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Document:
        """
        Get document by sequential integer index.

        Args:
            index: Integer index (0-based)

        Returns:
            Document at the given index

        Raises:
            IndexError: If index is out of range

        Example:
            doc = doc_store[0]  # Get first document
            doc = doc_store[-1]  # Get last document
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the total number of documents (same as count()).

        Returns:
            Number of documents

        Example:
            length = len(doc_store)
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
            filters: Optional metadata filters (if supported by implementation)

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
