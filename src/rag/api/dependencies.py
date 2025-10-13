from logging import Logger

import psycopg
import redis
from contextlib import contextmanager
from psycopg_pool import ConnectionPool
from fastapi.requests import Request

from rag.config import settings, Settings
from rag.logger import setup_logger
from rag.embeddings import BaseEmbedder

logger = setup_logger(__name__)

postgres_params = {
    "host": settings.postgres_host,
    "port": settings.postgres_port,
    "dbname": settings.postgres_db,
    "user": settings.postgres_user,
    "password": settings.postgres_password.get_secret_value(),
}
pool = ConnectionPool(
    min_size=1,
    max_size=10,
    kwargs=postgres_params,
)


def get_settings() -> Settings:
    return settings

def get_logger() -> Logger:
    return setup_logger(name=__name__)

@contextmanager
def get_postgres_conn():
    """Get PostgreSQL connection from pool"""
    with pool.connection() as conn:
        yield conn

def get_redis_client() -> redis.Redis:
    """Get Redis client"""
    client = redis.from_url(settings.redis_url)
    return client

def get_embedder(request: Request) -> BaseEmbedder:
    """
    Get embedder from app state (dependency injection).

    Returns BaseEmbedder interface - actual implementation
    depends on config (LocalEmbedder, OpenAIEmbedder, etc.)
    """
    return request.app.state.embedder