from rag.config import Settings
from rag.embeddings import BaseEmbedder
from rag.logger import setup_logger
from rag.storage.base import (
    BaseDocumentStore,
    BaseVectorStore,
    DocumentNotFoundError,
    ChunkNotFoundError,
)
from rag.storage.models import Document, SearchResult, make_chunk_id, parse_chunk_id
from rag.storage.document_stores import InMemoryDocumentStore
from rag.storage.vector_stores import FAISSVectorStore

logger = setup_logger(__name__)

__all__ = [
    # Base classes
    "BaseDocumentStore",
    "BaseVectorStore",
    # Exceptions
    "DocumentNotFoundError",
    "ChunkNotFoundError",
    # Models
    "Document",
    "SearchResult",
    "make_chunk_id",
    "parse_chunk_id",
    # Implementations
    "InMemoryDocumentStore",
    "FAISSVectorStore",
]

def create_doc_store(settings: Settings) -> BaseDocumentStore:
    """
    Create document store and auto-load from disk if path exists.

    Args:
        settings: Settings object with doc_store type and path

    Returns:
        Document store instance (loaded or empty)
    """
    from pathlib import Path
    from rag.config import PROJECT_ROOT

    if settings.doc_store == "inmemory":
        store = InMemoryDocumentStore()

        # Check if saved store exists and load it
        store_path = PROJECT_ROOT / settings.doc_store_path
        if store_path.exists():
            logger.info(f"Loading document store from {store_path}")
            store.load(str(store_path))
            logger.info(f"Loaded {len(store)} documents")
        else:
            logger.info(f"Creating new empty document store (no file at {store_path})")

        return store

    raise ValueError(f"Unknown Document Store: {settings.doc_store}")

def create_vector_store(settings: Settings, embedder: BaseEmbedder) -> BaseVectorStore:
    """
    Create vector store and auto-load from disk if path exists.

    Args:
        settings: Settings object with vec_store type, path, and embedding_dimension
        embedder: Embedder instance (unused, kept for backwards compatibility)

    Returns:
        Vector store instance (loaded or empty)
    """
    from pathlib import Path
    from rag.config import PROJECT_ROOT

    if settings.vec_store == "faiss":
        store = FAISSVectorStore(settings.embedding_dimension)

        # Check if saved store exists and load it
        store_path = PROJECT_ROOT / settings.vec_store_path
        index_file = Path(f"{store_path}.index")

        if index_file.exists():
            logger.info(f"Loading vector store from {store_path}")
            store.load(str(store_path))
            logger.info(f"Loaded {store.count()} vectors")
        else:
            logger.info(f"Creating new empty vector store (no file at {store_path})")

        return store

    raise ValueError(f"Unknown Vector Store: {settings.vec_store}")


