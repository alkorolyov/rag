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
from rag.storage.document_stores import InMemoryDocumentStore, PostgresDocumentStore
from rag.storage.vector_stores import FAISSVectorStore, PgvectorVectorStore

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
    "PostgresDocumentStore",
    "FAISSVectorStore",
    "PgvectorVectorStore",
]

def create_doc_store(settings: Settings) -> BaseDocumentStore:
    """
    Create document store.

    Args:
        settings: Settings object with doc_store type

    Returns:
        Document store instance
    """
    if settings.doc_store == "inmemory":
        from pathlib import Path
        from rag.config import PROJECT_ROOT

        store = InMemoryDocumentStore()
        store_path = PROJECT_ROOT / settings.doc_store_path

        if store_path.exists():
            logger.info(f"Loading document store from {store_path}")
            store.load(str(store_path))
            logger.info(f"Loaded {store.count_chunks()} chunks, {store.count_documents()} documents")
        else:
            logger.info(f"Creating new empty document store")

        return store

    if settings.doc_store == "postgres":
        store = PostgresDocumentStore(settings)
        logger.info(f"Connected to PostgreSQL: {store.count_chunks()} chunks, {store.count_documents()} documents")
        return store

    raise ValueError(f"Unknown doc_store: {settings.doc_store}")

def create_vector_store(settings: Settings, embedder: BaseEmbedder = None) -> BaseVectorStore:
    """
    Create vector store.

    Args:
        settings: Settings object with vec_store type and embedding_dimension
        embedder: Embedder instance (unused, kept for backwards compatibility)

    Returns:
        Vector store instance
    """
    if settings.vec_store == "faiss":
        from pathlib import Path
        from rag.config import PROJECT_ROOT

        store = FAISSVectorStore(settings.embedding_dimension)
        store_path = PROJECT_ROOT / settings.vec_store_path
        index_file = Path(f"{store_path}.index")

        if index_file.exists():
            logger.info(f"Loading vector store from {store_path}")
            store.load(str(store_path))
            logger.info(f"Loaded {store.count()} vectors")
        else:
            logger.info(f"Creating new empty vector store")

        return store

    if settings.vec_store == "pgvector":
        store = PgvectorVectorStore(settings)
        logger.info(f"Connected to pgvector: {store.count()} vectors")
        return store

    raise ValueError(f"Unknown vec_store: {settings.vec_store}")


