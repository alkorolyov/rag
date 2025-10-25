import datetime as dt

from sqlalchemy.dialects.postgresql import JSONB

from rag.config import settings
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = 'documents'

    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    meta = Column(JSONB, default={})
    created_at = Column(DateTime, nullable=False, default=lambda: dt.datetime.now(dt.UTC))

    __table_args__ = (
        Index(
            'idx_documents_metadata',
            'meta',
            postgresql_using='gin',
            postgresql_ops={'meta': 'jsonb_path_ops'}
        ),
    )

class ChunkModel(Base):
    __tablename__ = 'chunks'

    id = Column(String, primary_key=True)
    doc_id = Column(String, ForeignKey('documents.id'))
    chunk_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dimension))
    meta = Column(JSONB, default={})
    created_at = Column(DateTime, nullable=False, default=lambda: dt.datetime.now(dt.UTC))

    __table_args__ = (
        Index('idx_chunks_doc_id', 'doc_id'),
    )