from typing import List, Dict, Any

from rag.storage.base import BaseDocumentStore, ChunkNotFoundError
from rag.storage.models import Document, parse_chunk_id


class InMemoryDocumentStore(BaseDocumentStore):
    """
    Simple in-memory document store using a dictionary.

    This implementation stores documents in a Python dict and is suitable for:
    - Prototyping and development
    - Small to medium datasets (up to ~100K documents)
    - Testing

    For production with large datasets, consider PostgreSQL or other persistent stores.
    """

    def __init__(self):
        """Initialize an empty in-memory document store."""
        self._store: Dict[str, Document] = {}

    def add(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of chunk IDs that were added

        Raises:
            ValueError: If documents list is empty or contains invalid documents
        """
        if not documents:
            raise ValueError("Cannot add empty documents list")

        chunk_ids = []
        for doc in documents:
            if not doc.id:
                raise ValueError("Document must have an id")
            if not doc.text:
                raise ValueError(f"Document {doc.id} must have text content")

            chunk_id = str(doc.id)
            self._store[chunk_id] = doc
            chunk_ids.append(chunk_id)

        return chunk_ids

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
        if chunk_id not in self._store:
            raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")
        return self._store[chunk_id]

    def get_by_ids(self, chunk_ids: List[str]) -> List[Document]:
        """
        Get multiple documents by chunk IDs.

        This method is lenient and skips missing chunk IDs without raising
        an exception. The order of returned documents matches the input order.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            List of found Document objects (may be shorter than input if some missing)
        """
        documents = []
        for chunk_id in chunk_ids:
            if chunk_id in self._store:
                documents.append(self._store[chunk_id])
        return documents

    def get_by_doc_id(self, doc_id: str | int) -> List[Document]:
        """
        Get all chunks belonging to a parent document.

        This searches for chunks whose ID starts with "{doc_id}#" pattern.

        Args:
            doc_id: The parent document identifier

        Returns:
            List of Document chunks sorted by chunk index (empty list if none found)
        """
        doc_id_str = str(doc_id)
        matching_docs = []

        for chunk_id, doc in self._store.items():
            try:
                parsed_doc_id, chunk_index = parse_chunk_id(chunk_id)
                if parsed_doc_id == doc_id_str:
                    matching_docs.append((chunk_index, doc))
            except ValueError:
                # Not in chunk_id format, check if it matches directly
                if chunk_id.startswith(f"{doc_id_str}#"):
                    matching_docs.append((0, doc))

        # Sort by chunk index
        matching_docs.sort(key=lambda x: x[0])
        return [doc for _, doc in matching_docs]

    def get_parent_document(self, doc_id: str | int) -> Document | None:
        """
        Get the parent document (full document, not a chunk).

        Args:
            doc_id: The parent document identifier (without '#' symbol)

        Returns:
            The parent Document object, or None if not found
        """
        doc_id_str = str(doc_id)

        # Try direct lookup
        if doc_id_str in self._store:
            doc = self._store[doc_id_str]
            # Verify it's a parent (no '#' in ID)
            if "#" not in str(doc.id):
                return doc

        return None

    def filter_by_metadata(self, filters: Dict[str, Any]) -> List[Document]:
        """
        Filter documents by metadata fields.

        Performs exact matching on metadata fields. All filter conditions
        must match (AND logic).

        Args:
            filters: Dictionary of metadata field filters
                    Example: {"year": 2023, "source": "pubmed"}

        Returns:
            List of matching Documents (empty list if none match)
        """
        if not filters:
            return list(self._store.values())

        matching_docs = []
        for doc in self._store.values():
            # Check if all filter conditions match
            if all(
                key in doc.metadata and doc.metadata[key] == value
                for key, value in filters.items()
            ):
                matching_docs.append(doc)

        return matching_docs

    def delete(self, chunk_ids: List[str]) -> None:
        """
        Delete documents by chunk IDs.

        Args:
            chunk_ids: List of chunk identifiers to delete

        Raises:
            ChunkNotFoundError: If any chunk_id does not exist
        """
        # First check all exist
        missing = [cid for cid in chunk_ids if cid not in self._store]
        if missing:
            raise ChunkNotFoundError(
                f"Cannot delete: chunks not found: {missing}"
            )

        # Delete all
        for chunk_id in chunk_ids:
            del self._store[chunk_id]

    def count(self) -> int:
        """
        Get the total number of documents in the store.

        Returns:
            Number of documents
        """
        return len(self._store)

    def __iter__(self):
        """
        Iterate over all documents in the store.

        Yields:
            Document objects in insertion order
        """
        return iter(self._store.values())

    def __getitem__(self, index: int) -> Document:
        """
        Get document by sequential integer index.

        Args:
            index: Integer index (0-based), supports negative indexing

        Returns:
            Document at the given index

        Raises:
            IndexError: If index is out of range
        """
        docs_list = list(self._store.values())
        try:
            return docs_list[index]
        except IndexError:
            raise IndexError(f"Index {index} out of range for store with {len(docs_list)} documents")

    def __len__(self) -> int:
        """
        Get the total number of documents (same as count()).

        Returns:
            Number of documents
        """
        return len(self._store)

    def save(self, path: str) -> None:
        """
        Save the document store to disk using pickle.

        Args:
            path: File path to save to (e.g., "data/doc_store.pkl")
        """
        from pathlib import Path
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._store, f)

    def load(self, path: str) -> None:
        """
        Load the document store from disk.

        Args:
            path: File path to load from

        Raises:
            FileNotFoundError: If file does not exist
        """
        from pathlib import Path
        import pickle

        if not Path(path).exists():
            raise FileNotFoundError(f"Document store file not found: {path}")

        with open(path, "rb") as f:
            self._store = pickle.load(f)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(count={self.count()})"
