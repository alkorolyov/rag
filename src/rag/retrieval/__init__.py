from rag.config import Settings
from rag.retrieval.reranker import CrossEncoderReranker, BaseReranker


def create_reranker(settings: Settings) -> BaseReranker:
    return CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)