from rag.config import Settings
from rag.ingestion.chunker import BaseChunker, RecursiveChunker


def create_chunker(settings: Settings) -> BaseChunker:
    if settings.chunker_type == "recursive":
        return RecursiveChunker(settings.chunk_size, settings.chunk_overlap, settings.embedding_model)
