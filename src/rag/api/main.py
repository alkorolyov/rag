"""FastAPI application for RAG API."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import Response

from rag.config import settings
from rag.logger import setup_logger
from rag.api.routes import health, embed, query
from rag.retrieval import HybridRetriever, HybridConfig

logger = setup_logger(name=__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting RAG API...")
    logger.info(f"Environment: {settings.environment}")

    # Initialize hybrid retriever
    config = HybridConfig(
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        collection_name="rag-api",
    )
    app.state.retriever = HybridRetriever(config)
    logger.info(f"HybridRetriever initialized: {config.embedding_model}")

    yield

    logger.info("Stopping RAG API...")


app = FastAPI(
    title="RAG API",
    description="Biomedical RAG system with hybrid search",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(embed.router)
app.include_router(query.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
