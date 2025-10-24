import os
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_project_root() -> Path:
    """Find project root by searching for pyproject.toml"""
    # First check environment variable (production)
    if env_root := os.getenv("PROJECT_ROOT"):
        return Path(env_root)

    # Search upward for marker (development/notebooks)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback to current working directory
    return Path.cwd()

PROJECT_ROOT = get_project_root()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # general
    environment: str
    log_level: str
    api_host: str
    api_port: int

    # database
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: SecretStr

    # redis
    redis_url: str
    redis_appendonly: bool
    redis_maxmemory: str
    redis_maxmemory_policy: str

    # embeddings
    embedding_provider: str
    embedding_model: str
    embedding_device: str
    k: int

    # chunking
    chunker_type: str  # e.g., "recursive"
    chunk_size: int
    chunk_overlap: int

    # reranker
    reranker_model: str
    reranker_device: str
    top_k: int

    # llm
    llm_type: str
    llm_device: str
    llm_model: str
    llm_temperature: float
    max_tokens: int

    # api keys
    openai_api_key: SecretStr
    aws_access_key_id: SecretStr
    aws_secret_access_key: SecretStr

    # dvc
    dvc_s3_bucket: str

    # mlflow
    mlflow_tracking_uri: str

settings = Settings()