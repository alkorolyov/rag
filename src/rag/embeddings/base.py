import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding models.

    Provides a unified interface for different embedding providers
    (sentence-transformers, OpenAI, Cohere, etc.).

    All embeddings are returned as numpy arrays with float32 dtype
    for memory efficiency and compatibility with FAISS.
    """

    @abstractmethod
    def embed_text(self, text: str) -> NDArray[np.float32]:
        """
        Embed a single text into a vector representation.

        Args:
            text: Input text to embed

        Returns:
            1D numpy array of shape (dimension,) with dtype float32

        Example:
            >>> embedder = LocalEmbedder()
            >>> vector = embedder.embed_text("myocardial infarction")
            >>> vector.shape
            (384,)
        """
        ...

    @abstractmethod
    def embed_batch(self, batch: List[str], batch_size: int = 32) -> NDArray[np.float32]:
        """
        Embed multiple texts into vector representations (more efficient than calling embed_text repeatedly).

        Args:
            batch: List of texts to embed
            batch_size: Batch size

        Returns:
            2D numpy array of shape (batch_size, dimension) with dtype float32

        Example:
            >>> embedder = LocalEmbedder()
            >>> vectors = embedder.embed_batch(["heart attack", "stroke"])
            >>> vectors.shape
            (2, 384)
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.

        Returns:
            Integer dimension (e.g., 384 for all-MiniLM-L6-v2,
            1536 for OpenAI text-embedding-3-small)
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(\"{self.model_name}\", dim={self.dimension})"
