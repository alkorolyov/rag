from typing import List, Optional, Dict, Any, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from rag.logger import setup_logger
from rag.storage.base import BaseDocumentStore, ChunkNotFoundError, DocumentNotFoundError
from rag.storage.models import Document, parse_chunk_id
from rag.storage.db_models import Base, ChunkModel, DocumentModel
from rag.config import Settings

logger = setup_logger(__name__)


class PostgresDocumentStore(BaseDocumentStore):
    """PostgreSQL-backed document store with connection pooling."""

    def __init__(self, settings: Settings):
        connection_string = (
            f"postgresql+psycopg://{settings.postgres_user}:"
            f"{settings.postgres_password.get_secret_value()}@"
            f"{settings.postgres_host}:{settings.postgres_port}/"
            f"{settings.postgres_db}"
        )

        self.engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        Base.metadata.create_all(self.engine)
        logger.info("PostgresDocumentStore initialized")

    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _chunk_to_pydantic(chunk: ChunkModel) -> Document:
        return Document(
            id=chunk.id,
            text=chunk.text,
            meta=chunk.meta or {},
            doc_type='chunk',
            score=0.0,
        )

    @staticmethod
    def _doc_to_pydantic(doc: DocumentModel) -> Document:
        return Document(
            id=doc.id,
            text=doc.text,
            meta=doc.meta or {},
            doc_type='parent',
            score=0.0,
        )

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
        with self.get_session() as session:
            for doc in documents:
                db_doc = DocumentModel(
                    id=str(doc.id),
                    text=doc.text,
                    meta=doc.meta or {},
                )
                session.merge(db_doc)
                doc_ids.append(str(doc.id))

        return doc_ids

    def add_chunks(self, chunks: List[Document]) -> List[str]:
        """Add chunks."""
        if not chunks:
            return []

        # Business logic validation only
        invalid = [c.id for c in chunks if c.doc_type != "chunk"]
        if invalid:
            raise ValueError(f"Chunks must have doc_type='chunk': {invalid}")

        chunk_ids = []
        with self.get_session() as session:
            for chunk in chunks:
                doc_id, chunk_index = parse_chunk_id(str(chunk.id))

                db_chunk = ChunkModel(
                    id=str(chunk.id),
                    doc_id=doc_id,
                    chunk_id=chunk_index,
                    text=chunk.text,
                    meta=chunk.meta or {},
                )
                session.merge(db_chunk)
                chunk_ids.append(str(chunk.id))

        return chunk_ids

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks (idempotent - no error if some don't exist)."""
        if not chunk_ids:
            return

        with self.get_session() as session:
            session.query(ChunkModel).filter(
                ChunkModel.id.in_(chunk_ids)
            ).delete(synchronize_session=False)

    def delete_document(self, doc_id: str) -> None:
        """Delete document and its chunks."""
        with self.get_session() as session:
            # Delete chunks first
            session.query(ChunkModel).filter(ChunkModel.doc_id == str(doc_id)).delete()
            # Delete document
            session.query(DocumentModel).filter(DocumentModel.id == str(doc_id)).delete()

    # ===== Read Operations - Chunks =====
    def get_chunk(self, chunk_id: str) -> Document:
        """Get single chunk. Raises ChunkNotFoundError if not found."""
        with self.get_session() as session:
            chunk = session.get(ChunkModel, chunk_id)
            if not chunk:
                raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")
            return self._chunk_to_pydantic(chunk)

    def get_chunks(self, chunk_ids: List[str]) -> List[Document]:
        """Get multiple chunks."""
        if not chunk_ids:
            return []

        with self.get_session() as session:
            chunks = session.query(ChunkModel).filter(
                ChunkModel.id.in_(chunk_ids)
            ).all()
            return [self._chunk_to_pydantic(c) for c in chunks]

    def get_document_chunks(self, doc_id: str) -> List[Document]:
        """Get all chunks for a document."""
        with self.get_session() as session:
            chunks = session.query(ChunkModel).filter(
                ChunkModel.doc_id == str(doc_id)
            ).order_by(ChunkModel.chunk_id).all()
            return [self._chunk_to_pydantic(c) for c in chunks]

    def filter_chunks(self, filters: Dict[str, Any]) -> List[Document]:
        """Filter chunks by metadata (not implemented)."""
        # TODO: Implement JSONB filtering when needed
        return []

    def iter_chunks(self, batch_size: int = 100):
        """Iterate over batches of chunks."""
        with self.get_session() as session:
            offset = 0
            while True:
                batch = session.query(ChunkModel).offset(offset).limit(batch_size).all()
                if not batch:
                    break
                yield [self._chunk_to_pydantic(c) for c in batch]
                offset += batch_size

    def get_chunks_paginated(self, offset: int = 0, limit: int = 100) -> List[Document]:
        """Get chunks with pagination."""
        with self.get_session() as session:
            chunks = session.query(ChunkModel).offset(offset).limit(limit).all()
            return [self._chunk_to_pydantic(c) for c in chunks]

    # ===== Read Operations - Documents =====
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get parent document."""
        with self.get_session() as session:
            doc = session.get(DocumentModel, str(doc_id))
            return self._doc_to_pydantic(doc) if doc else None

    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """Get multiple documents."""
        if not doc_ids:
            return []

        with self.get_session() as session:
            docs = session.query(DocumentModel).filter(
                DocumentModel.id.in_(doc_ids)
            ).all()
            return [self._doc_to_pydantic(d) for d in docs]

    # ===== Stats =====
    def count_chunks(self) -> int:
        """Total chunks."""
        with self.get_session() as session:
            return session.query(func.count(ChunkModel.id)).scalar()

    def count_documents(self) -> int:
        """Total documents."""
        with self.get_session() as session:
            return session.query(func.count(DocumentModel.id)).scalar()

    def close(self):
        """Close connection pool."""
        self.engine.dispose()
        logger.info("PostgresDocumentStore closed")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"documents={self.count_documents()}, chunks={self.count_chunks()})"
        )
