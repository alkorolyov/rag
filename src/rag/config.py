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
    redis_ttl: int
    redis_semantic_distance: float
    redis_appendonly: bool
    redis_maxmemory: str
    redis_maxmemory_policy: str

    # document store
    doc_store: str
    doc_store_path: str = "data/doc_store.pkl"  # Path to save/load document store

    # vector store
    vec_store: str
    vec_store_path: str = "data/vec_store"  # Path to save/load vector store (without extension)

    # embeddings
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
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
    llm_max_tokens: int

    # api keys
    openai_api_key: SecretStr
    aws_access_key_id: SecretStr
    aws_secret_access_key: SecretStr

    # dvc
    dvc_s3_bucket: str

    # mlflow
    mlflow_tracking_uri: str

    # huggingface datasets
    hf_datasets_cache: str = ".cache"

settings = Settings()

# Auto-configure HuggingFace datasets cache with absolute path
cache_path = Path(settings.hf_datasets_cache)
if not cache_path.is_absolute():
    cache_path = PROJECT_ROOT / cache_path
cache_path = cache_path.resolve()  # Always resolve to absolute path
os.environ['HF_DATASETS_CACHE'] = str(cache_path)