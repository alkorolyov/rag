from typing import List, Dict, Any, Optional

from rag.storage.base import BaseDocumentStore, ChunkNotFoundError, DocumentNotFoundError
from rag.storage.models import Document, parse_chunk_id


class InMemoryDocumentStore(BaseDocumentStore):
    """
    Simple in-memory document store using dictionaries.

    Stores parent documents and chunks separately for clarity.
    Suitable for prototyping, development, and testing.
    """

    def __init__(self):
        """Initialize empty in-memory stores."""
        self._documents: Dict[str, Document] = {}  # Parent documents
        self._chunks: Dict[str, Document] = {}      # Chunks

    # ===== Write Operations =====
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add parent documents."""
        if not documents:
            return []

        # Business logic validation only
        invalid = [d.id for d in documents if d.doc_type != "parent"]
        if invalid:
            raise ValueError(f"Documents must have doc_type='parent': {invalid}")

        doc_ids = []
        for doc in documents:
            doc_id = str(doc.id)
            self._documents[doc_id] = doc
            doc_ids.append(doc_id)

        return doc_ids

    def add_chunks(self, chunks: List[Document]) -> List[str]:
        """Add document chunks."""
        if not chunks:
            return []

        # Business logic validation only
        invalid = [c.id for c in chunks if c.doc_type != "chunk"]
        if invalid:
            raise ValueError(f"Chunks must have doc_type='chunk': {invalid}")

        chunk_ids = []
        for chunk in chunks:
            chunk_id = str(chunk.id)
            self._chunks[chunk_id] = chunk
            chunk_ids.append(chunk_id)

        return chunk_ids

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs (idempotent - no error if some don't exist)."""
        if not chunk_ids:
            return

        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)

    def delete_document(self, doc_id: str) -> None:
        """Delete parent document and all its chunks (idempotent)."""
        doc_id = str(doc_id)

        # Delete parent
        self._documents.pop(doc_id, None)

        # Delete all chunks
        chunks_to_delete = [
            chunk_id for chunk_id in list(self._chunks.keys())
            if chunk_id.startswith(f"{doc_id}#")
        ]
        for chunk_id in chunks_to_delete:
            del self._chunks[chunk_id]

    # ===== Read Operations - Chunks =====
    def get_chunk(self, chunk_id: str) -> Document:
        """Get single chunk by ID. Raises ChunkNotFoundError if not found."""
        if chunk_id not in self._chunks:
            raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")
        return self._chunks[chunk_id]

    def get_chunks(self, chunk_ids: List[str]) -> List[Document]:
        """Get multiple chunks."""
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_document_chunks(self, doc_id: str) -> List[Document]:
        """Get all chunks for a document."""
        doc_id_str = str(doc_id)
        matching_docs = []

        for chunk_id, chunk in self._chunks.items():
            try:
                parsed_doc_id, chunk_index = parse_chunk_id(chunk_id)
                if parsed_doc_id == doc_id_str:
                    matching_docs.append((chunk_index, chunk))
            except ValueError:
                if chunk_id.startswith(f"{doc_id_str}#"):
                    matching_docs.append((0, chunk))

        matching_docs.sort(key=lambda x: x[0])
        return [doc for _, doc in matching_docs]

    def filter_chunks(self, filters: Dict[str, Any]) -> List[Document]:
        """Filter chunks by metadata."""
        if not filters:
            return list(self._chunks.values())

        matching = []
        for chunk in self._chunks.values():
            if all(
                key in chunk.meta and chunk.meta[key] == value
                for key, value in filters.items()
            ):
                matching.append(chunk)
        return matching

    def iter_chunks(self, batch_size: int = 100):
        """Iterate over batches of chunks."""
        chunks = list(self._chunks.values())
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def get_chunks_paginated(self, offset: int = 0, limit: int = 100) -> List[Document]:
        """Get chunks with pagination."""
        chunks = list(self._chunks.values())
        return chunks[offset:offset + limit]

    # ===== Read Operations - Documents =====
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get parent document."""
        return self._documents.get(str(doc_id))

    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """Get multiple parent documents."""
        return [self._documents[str(did)] for did in doc_ids if str(did) in self._documents]

    # ===== Stats =====
    def count_chunks(self) -> int:
        """Total chunks."""
        return len(self._chunks)

    def count_documents(self) -> int:
        """Total parent documents."""
        return len(self._documents)

    # ===== Persistence =====
    def save(self, path: str) -> None:
        """Save to disk."""
        from pathlib import Path
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"documents": self._documents, "chunks": self._chunks}, f)

    def load(self, path: str) -> None:
        """Load from disk."""
        from pathlib import Path
        import pickle

        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            self._documents = data["documents"]
            self._chunks = data["chunks"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"documents={self.count_documents()}, chunks={self.count_chunks()})"
        )
