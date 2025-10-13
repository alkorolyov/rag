from .base import BaseEmbedder
from .local_embedder import LocalEmbedder
from rag.config import Settings


def create_embedder(settings: Settings) -> BaseEmbedder:
    if settings.embedding_provider == "sentence-transformers":
        return LocalEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device
        )
    elif settings.embedding_provider == "openai":
        raise NotImplementedError("OPENAI not implemented.")
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}. Supported providers: 'sentence-transformers', 'openai'")

__all__ = ["create_embedder", "BaseEmbedder", "LocalEmbedder"]