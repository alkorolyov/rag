from typing import Optional, List

import numpy as np
from numpy.typing import NDArray
import torch

from sentence_transformers import SentenceTransformer
from rag.embeddings.base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """
    Local sentence-transformers embedding model.

    Uses the sentence-transformers library to generate embeddings locally
    (no API calls required). Supports any model from HuggingFace Hub.

    All embeddings are normalized to unit vectors (L2 norm = 1.0) for
    efficient cosine similarity computation via dot product.

    Args:
        model_name: HuggingFace model identifier (e.g., 'all-MiniLM-L6-v2')
        **kwargs: Additional arguments passed to SentenceTransformer
            (e.g., device='cuda', cache_folder='/path/to/cache')

    Example:
        >>> embedder = LocalEmbedder("all-MiniLM-L6-v2")
        >>> vector = embedder.embed_text("myocardial infarction")
        >>> vector.shape
        (384,)
        >>> import numpy as np
        >>> np.linalg.norm(vector)  # Should be 1.0 (normalized)
        1.0
    """

    def __init__(self, model_name: str, **kwargs):
        self._model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model.eval()  # Ensure model is in evaluation mode

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """
        Embed a single text into a normalized vector representation.

        Args:
            text: Input text to embed

        Returns:
            1D numpy array of shape (dimension,) with dtype float32,
            normalized to unit length (L2 norm = 1.0)

        Example:
            >>> embedder = LocalEmbedder("all-MiniLM-L6-v2")
            >>> vector = embedder.embed_text("heart attack")
            >>> vector.shape
            (384,)
        """
        result = self.model.encode(text, normalize_embeddings=True)
        return result

    def embed_batch(self, batch: List[str], batch_size: int = 32) -> NDArray[np.float32]:
        """
        Embed multiple texts into normalized vector representations.

        More efficient than calling embed_text repeatedly due to batched
        processing on CPU/GPU.

        Args:
            batch: List of texts to embed

        Returns:
            2D numpy array of shape (batch_size, dimension) with dtype float32,
            each row normalized to unit length (L2 norm = 1.0)

        Example:
            >>> embedder = LocalEmbedder("all-MiniLM-L6-v2")
            >>> vectors = embedder.embed_batch(["heart attack", "stroke"])
            >>> vectors.shape
            (2, 384)
        """
        result = self.model.encode(batch, normalize_embeddings=True, batch_size=batch_size)
        return result

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.

        Returns:
            Integer dimension (384 for all-MiniLM-L6-v2,
            768 for all-mpnet-base-v2, etc.)

        Example:
            >>> embedder = LocalEmbedder("all-MiniLM-L6-v2")
            >>> embedder.dimension
            384
        """
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name
