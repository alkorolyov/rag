from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import Response

from rag.config import settings
from rag.embeddings import create_embedder
from rag.generation import create_llm
from rag.logger import setup_logger
from rag.api.routes import health, embed

logger = setup_logger(name=__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG API ...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API: https://{settings.api_host}:{settings.api_port}")

    app.state.embedder = create_embedder(settings)
    logger.info(f"Embedding model loaded: {settings.embedding_model}")

    app.state.llm_model = create_llm(settings)
    logger.info(f"LLM model loaded: {settings.llm_model}")

    yield

    logger.info("Stopping RAG API ...")


app = FastAPI(
    title="RAG General API",
    description="Production-grade RAG system for general purposes",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(embed.router)

@app.get('/')
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the RAG General API!",
        "version": "0.1.0",
        "docs": "/docs",
    }

@app.get('/favicon.ico')
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)