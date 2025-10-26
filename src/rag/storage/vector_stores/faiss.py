import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from numpy.typing import NDArray

from rag.storage.base import BaseVectorStore, ChunkNotFoundError
from rag.storage.models import SearchResult


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store for efficient similarity search.

    This implementation uses Facebook AI Similarity Search (FAISS) for vector indexing
    and search. It supports multiple index types and provides disk persistence.

    Suitable for:
    - Development and prototyping
    - Medium to large datasets (millions of vectors)
    - Local/on-premise deployment

    For cloud-native or distributed deployments, consider pgvector, Pinecone, or Qdrant.
    """

    SUPPORTED_INDEXES = {
        "flat_ip": faiss.IndexFlatIP,      # Inner product (cosine similarity)
        "flat_l2": faiss.IndexFlatL2,      # L2 distance (Euclidean)
        "ivf_flat": faiss.IndexIVFFlat,    # Inverted file index (faster, approximate)
    }

    def __init__(self, dimension: int, index_type: str = "flat_ip"):
        """
        Initialize a FAISS vector store.

        Args:
            dimension: Dimensionality of embedding vectors
            index_type: Type of FAISS index to use (default: "flat_ip")
                       Options: "flat_ip", "flat_l2", "ivf_flat"

        Raises:
            ValueError: If index_type is not supported
        """
        if index_type not in self.SUPPORTED_INDEXES:
            raise ValueError(
                f"Invalid index type '{index_type}'. "
                f"Supported types: {list(self.SUPPORTED_INDEXES.keys())}"
            )

        self.dimension = dimension
        self.index_type = index_type

        # Mappings between FAISS internal IDs and chunk IDs
        self._chunk_id_to_faiss: Dict[str, int] = {}
        self._faiss_to_chunk_id: Dict[int, str] = {}

        # Create FAISS index
        base_index = self.SUPPORTED_INDEXES[index_type](dimension)
        self.index = faiss.IndexIDMap(base_index)

    def add(self, embeddings: NDArray[np.float32], chunk_ids: List[str]) -> None:
        """
        Add embeddings to the vector store.

        Args:
            chunk_ids: List of chunk identifiers
            embeddings: Numpy array of shape (n, dimension) with embedding vectors

        Raises:
            ValueError: If chunk_ids and embeddings lengths don't match
            ValueError: If embeddings have wrong dimension
            ValueError: If chunk_ids contains duplicates already in the store
        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunk_ids)} chunk_ids vs "
                f"{len(embeddings)} embeddings"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        # Check for duplicates
        duplicates = [cid for cid in chunk_ids if cid in self._chunk_id_to_faiss]
        if duplicates:
            raise ValueError(f"Chunk IDs already exist in store: {duplicates[:5]}")

        # Generate FAISS internal IDs
        n = self.index.ntotal
        faiss_ids = np.arange(n, n + len(chunk_ids), dtype=np.int64)

        # Add to FAISS index
        self.index.add_with_ids(embeddings, faiss_ids)

        # Update mappings
        for faiss_id, chunk_id in zip(faiss_ids, chunk_ids):
            self._chunk_id_to_faiss[chunk_id] = int(faiss_id)
            self._faiss_to_chunk_id[int(faiss_id)] = chunk_id

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return
            filters: Optional meta filters (not supported by FAISS, ignored)

        Returns:
            List of SearchResult objects sorted by score (highest first)

        Note:
            FAISS does not support native meta filtering. For filtered search,
            retrieve more results (e.g., k*10) and filter in DocumentStore.
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[1]}"
            )

        # Perform search
        k_actual = min(k, self.index.ntotal)
        if k_actual == 0:
            return []

        distances, indices = self.index.search(query_embedding, k_actual)

        # Convert to SearchResult objects
        results = []
        for distance, faiss_id in zip(distances.flatten(), indices.flatten()):
            # FAISS returns -1 for unfilled results
            if faiss_id == -1:
                continue

            faiss_id_int = int(faiss_id)
            if faiss_id_int not in self._faiss_to_chunk_id:
                continue

            chunk_id = self._faiss_to_chunk_id[faiss_id_int]
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=float(distance),
                    meta=None
                )
            )

        return results

    def delete(self, chunk_ids: List[str]) -> None:
        """
        Delete vectors by chunk IDs (idempotent - no error if some don't exist).

        Args:
            chunk_ids: List of chunk identifiers to delete
        """
        if not chunk_ids:
            return

        # Filter to only existing chunks
        existing_chunks = [cid for cid in chunk_ids if cid in self._chunk_id_to_faiss]
        if not existing_chunks:
            return

        # Get FAISS IDs
        faiss_ids = [self._chunk_id_to_faiss[cid] for cid in existing_chunks]

        # Remove from FAISS index
        faiss_ids_selector = faiss.IDSelectorArray(np.array(faiss_ids, dtype=np.int64))
        self.index.remove_ids(faiss_ids_selector)

        # Update mappings
        for chunk_id, faiss_id in zip(existing_chunks, faiss_ids):
            del self._chunk_id_to_faiss[chunk_id]
            del self._faiss_to_chunk_id[faiss_id]

    def count(self) -> int:
        """
        Get the total number of vectors in the store.

        Returns:
            Number of vectors
        """
        return self.index.ntotal

    def save(self, path: str) -> None:
        """
        Save the vector store to disk.

        Creates two files:
        - {path}.index: FAISS index
        - {path}.mappings: Chunk ID mappings

        Args:
            path: Base path for saving (without extension)
        """
        # Create parent directory
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")

        # Save mappings
        with open(f"{path}.mappings", "wb") as f:
            pickle.dump(
                {
                    "chunk_id_to_faiss": self._chunk_id_to_faiss,
                    "faiss_to_chunk_id": self._faiss_to_chunk_id,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                },
                f
            )

    def load(self, path: str) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Base path for loading (without extension)

        Raises:
            FileNotFoundError: If index or mappings file does not exist
        """
        index_path = f"{path}.index"
        mappings_path = f"{path}.mappings"

        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not Path(mappings_path).exists():
            raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load mappings
        with open(mappings_path, "rb") as f:
            data = pickle.load(f)
            self._chunk_id_to_faiss = data["chunk_id_to_faiss"]
            self._faiss_to_chunk_id = data["faiss_to_chunk_id"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"count={self.count()}, "
            f"dimension={self.dimension}, "
            f"index_type={self.index_type})"
        )
