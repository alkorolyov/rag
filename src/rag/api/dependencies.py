"""FastAPI dependencies for RAG API."""

from fastapi.requests import Request

from rag.config import settings, Settings
from rag.retrieval import HybridRetriever


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def get_retriever(request: Request) -> HybridRetriever:
    """Get HybridRetriever from app state."""
    return request.app.state.retriever
