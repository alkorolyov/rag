"""Text chunking utilities for document preprocessing."""

from abc import ABC, abstractmethod
from typing import List

from datasets import Dataset, disable_progress_bar, enable_progress_bar
from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter
from tqdm import tqdm

from rag.models import make_chunk_id
from rag.utils import get_token_counter


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        ...

    def chunk_dataset(
        self,
        dataset: Dataset,
        text_col: str = "text",
        id_col: str = "id",
    ) -> Dataset:
        """Chunk a HuggingFace Dataset."""

        def chunk_generator():
            for doc in tqdm(dataset, desc="Chunking"):
                text = doc[text_col]
                doc_id = doc[id_col]
                chunks = self.chunk_text(text)
                for i, chunk in enumerate(chunks):
                    yield {
                        "text": chunk,
                        "doc_id": doc_id,
                        "chunk_id": make_chunk_id(doc_id, i),
                    }

        disable_progress_bar()
        result = Dataset.from_generator(chunk_generator)
        enable_progress_bar()
        return result


class RecursiveChunker(BaseChunker):
    """Token-aware recursive text chunker."""

    def __init__(self, chunk_size: int, chunk_overlap: int, embedder_model: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=get_token_counter(embedder_model),
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.chunker.split_text(text)


class SentenceChunker(BaseChunker):
    """Sentence-aware chunker using spaCy."""

    def __init__(self, chunk_size: int, chunk_overlap: int, embedder_model: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = SpacyTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=get_token_counter(embedder_model),
            pipeline="en_core_web_sm",
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.chunker.split_text(text)
