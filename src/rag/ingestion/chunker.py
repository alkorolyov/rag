from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, List

from datasets import Dataset, disable_progress_bar, enable_progress_bar
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from rag.embeddings import BaseEmbedder
from rag.storage.models import Document, make_chunk_id
from rag.utils import get_token_counter


class BaseChunker(ABC):
    @abstractmethod
    def chunk_text(
            self,
            text: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def chunk_dataset(
            self,
            dataset: Dataset,
            text_col: str = "text",
            id_col: str = "id",
            metadata: Optional[Dict[str, Any]] = None,
    ):
        ...


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int, embedder_model: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=get_token_counter(embedder_model),
        )

    def chunk_text(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        results = []
        chunks = self.chunker.split_text(text)
        for i, chunk in enumerate(chunks):
            results.append({
                "text": chunk,
                "chunk_id": i,
            })
        return results

    def chunk_dataset(
            self,
            dataset: Dataset,
            text_col: str = "text",
            id_col: str = "id",
            **kwargs,
    ) -> Dataset:
        def chunk_generator():
            for doc in tqdm(dataset, desc='Chunking'):
                text = doc[text_col]
                doc_id = doc[id_col]
                chunks = self.chunker.split_text(text)
                for i, chunk in enumerate(chunks):
                    yield {
                        'text': chunk,
                        'doc_id': doc_id,
                        'chunk_id': make_chunk_id(doc_id, i),
                    }

        # Disable built-in progress bar
        disable_progress_bar()

        result = Dataset.from_generator(
            chunk_generator,
        )

        # Re-enable for other operations
        enable_progress_bar()
        return result

    def chunk_docs(self, docs: List[Document], text_col: str = "text", id_col: str = "id",) -> List[Dict[str, Any]]:
        results = []
        for doc in tqdm(docs, desc='Chunking'):
            doc_id = doc[id_col]
            chunks = self.chunker.split_text(doc[text_col])
            for i, chunk in enumerate(chunks):
                results.append({
                    'text': chunk,
                    'doc_id': doc_id,
                    'chunk_id': make_chunk_id(doc_id, i),
                })
        return results




