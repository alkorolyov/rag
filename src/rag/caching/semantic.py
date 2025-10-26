import json
from typing import List, Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation
from langchain_redis import RedisSemanticCache

from rag.config import Settings
from rag.embeddings import create_embedder


class LangChainEmbedderAdapter(Embeddings):
    """Adapter to make your BaseEmbedder work with LangChain"""

    def __init__(self, base_embedder):
        self.embedder = base_embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = self.embedder.embed_texts(texts)  # Returns (n, dim) numpy array
        return embeddings.tolist()  # Convert to list of lists

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.embedder.embed_text(text)  # Returns (dim,) numpy array
        return embedding.tolist()  # Convert to list


class SemanticCache:
    def __init__(self, settings: Settings):
        lang_chain_embedder = LangChainEmbedderAdapter(create_embedder(settings))
        self.cache = RedisSemanticCache(
            embeddings=lang_chain_embedder,
            redis_url=settings.redis_url,
            distance_threshold=settings.redis_semantic_distance,
        )
        self.llm_string = settings.llm_model + str(settings.llm_temperature)

    def get(self, question: str) -> Optional[Dict[str, any]]:
        """
        Retrieve cached answer with document IDs.

        Returns:
            Dict with keys:
                - answer: str (the LLM response)
                - doc_ids: List[str] (chunk IDs for source documents)
            or None if cache miss
        """
        results = self.cache.lookup(prompt=question, llm_string=self.llm_string)
        if results is not None and len(results) > 0:
            cached_text = results[0].text
            try:
                data = json.loads(cached_text)
                return data
            except json.JSONDecodeError:
                # Fallback for old cache entries (just answer text)
                return {"answer": cached_text, "doc_ids": []}
        return None

    def set(self, question: str, answer: str, doc_ids: List[str]) -> None:
        """
        Store answer with source document IDs.

        Args:
            question: User query
            answer: LLM-generated answer
            doc_ids: List of chunk_ids from reranked documents
        """
        cache_data = {
            "answer": answer,
            "doc_ids": doc_ids
        }
        # Encode as JSON and store in Generation
        cache_text = json.dumps(cache_data)
        self.cache.update(
            prompt=question,
            llm_string=self.llm_string,
            return_val=[Generation(text=cache_text)]
        )
