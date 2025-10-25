from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any

from sentence_transformers import CrossEncoder

from rag.storage.models import Document


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        ...

    # @abstractmethod
    # def rerank_batch(self, queries: List[str], documents: List[List[str]]) -> List[List[str]]:
    #     ...

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str, device='cuda') -> None:
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: List[Document], batch_size: int = 32) -> List[Document]:
        pairs = [[query, doc.text] for doc in documents]
        scores = self.model.predict(pairs, batch_size)

        for doc, score in zip(documents, scores):
            doc.score = float(score)

        ranked = sorted(documents, key=lambda doc: doc.score, reverse=True)
        return ranked