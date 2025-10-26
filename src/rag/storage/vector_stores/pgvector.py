from typing import List, Dict, Any, Optional

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from rag.storage.base import BaseVectorStore, ChunkNotFoundError
from rag.storage.models import SearchResult
from rag.storage.db_models import Base, ChunkModel
from rag.config import Settings
from rag.logger import setup_logger

logger = setup_logger(__name__)


class PgvectorVectorStore(BaseVectorStore):
    """
    pgvector-based vector store using PostgreSQL.

    Stores embeddings directly in the chunks table using pgvector extension.
    Supports cosine similarity and L2 distance operators.

    Suitable for:
    - Production deployments
    - Integration with existing PostgreSQL infrastructure
    - Cost-effective vector search (79% cheaper than managed services)
    """

    def __init__(self, settings: Settings, distance: str = "cosine"):
        """
        Initialize pgvector vector store.

        Args:
            settings: Settings object with database connection info
            distance: Distance metric to use ("cosine" or "l2")

        Raises:
            ValueError: If distance metric is not supported
        """
        if distance not in ["cosine", "l2"]:
            raise ValueError(f"Invalid distance: {distance}. Use 'cosine' or 'l2'")

        self.distance = distance
        self.dimension = settings.embedding_dimension

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
        logger.info(f"PgvectorVectorStore initialized ({distance} distance)")

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

    def add(self, embeddings: NDArray[np.float32], chunk_ids: List[str]) -> None:
        """
        Add embeddings to the vector store.

        Updates the embedding column for existing chunks in the database.

        Args:
            embeddings: Numpy array of shape (n, dimension)
            chunk_ids: List of chunk identifiers

        Raises:
            ValueError: If lengths don't match or dimension is wrong
            ChunkNotFoundError: If chunk_ids don't exist in database
        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunk_ids)} chunk_ids vs {len(embeddings)} embeddings"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        with self.get_session() as session:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                chunk = session.get(ChunkModel, chunk_id)
                if not chunk:
                    raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")

                # Update embedding
                chunk.embedding = embedding.tolist()

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors using pgvector operators.

        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return
            filters: Optional metadata filters (not implemented yet)

        Returns:
            List of SearchResult objects sorted by score (best first)
        """
        # Ensure query is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.flatten()

        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[0]}"
            )

        # Choose distance operator
        if self.distance == "cosine":
            # Cosine distance: <=>
            operator = "<=>"
        else:
            # L2 distance: <->
            operator = "<->"

        # Convert to list for PostgreSQL
        query_list = query_embedding.tolist()

        with self.get_session() as session:
            # Raw SQL query with pgvector operator
            query = text(f"""
                SELECT id, embedding {operator} :query_embedding AS distance
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY distance
                LIMIT :k
            """)

            result = session.execute(
                query,
                {"query_embedding": str(query_list), "k": k}
            )

            # Convert to SearchResult
            results = []
            for row in result:
                chunk_id, distance = row
                # Convert distance to similarity score (higher is better)
                # For cosine: distance is 0-2, convert to similarity 1-0
                # For L2: smaller is better, use negative distance
                if self.distance == "cosine":
                    score = 1.0 - (float(distance) / 2.0)
                else:
                    score = -float(distance)

                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        score=score,
                        meta=None
                    )
                )

            return results

    def delete(self, chunk_ids: List[str]) -> None:
        """
        Delete vectors by setting embedding to NULL (idempotent).

        Args:
            chunk_ids: List of chunk identifiers
        """
        if not chunk_ids:
            return

        with self.get_session() as session:
            for chunk_id in chunk_ids:
                chunk = session.get(ChunkModel, chunk_id)
                if chunk:
                    chunk.embedding = None

    def count(self) -> int:
        """
        Get total number of vectors (chunks with embeddings).

        Returns:
            Number of chunks with non-NULL embeddings
        """
        with self.get_session() as session:
            result = session.execute(
                text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            )
            return result.scalar()

    def close(self):
        """Close connection pool."""
        self.engine.dispose()
        logger.info("PgvectorVectorStore closed")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"count={self.count()}, "
            f"dimension={self.dimension}, "
            f"distance={self.distance})"
        )
