from rag.caching.semantic import SemanticCache
from rag.config import Settings, settings
from rag.embeddings import BaseEmbedder
from rag.generation.base import BaseLLM
from rag.prompts import get_user_prompt, SYSTEM_PROMPT
from rag.retrieval import BaseReranker
from rag.storage import BaseVectorStore, BaseDocumentStore


class RAGPipeline:
    def __init__(self,
                 doc_store: BaseDocumentStore, vector_store: BaseVectorStore,
                 embedder: BaseEmbedder, reranker: BaseReranker, llm: BaseLLM, settings: Settings,):
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
        self.settings = settings
        self.cache = SemanticCache(settings)

    def query(self, question: str) -> tuple[str, list]:
        """
        Query the RAG pipeline.

        Args:
            question: User's question

        Returns:
            Tuple of (answer, source_documents)
        """
        # 0. Check semantic cache
        cached_data = self.cache.get(question)
        if cached_data is not None:
            answer = cached_data["answer"]
            chunk_ids = cached_data["doc_ids"]

            # Re-fetch documents from store to provide sources
            reranked_chunks = [self.doc_store.get_chunk(doc_id) for doc_id in chunk_ids]

            return answer, reranked_chunks

        # 1. Embed query
        q_emb = self.embedder.embed_text(question)

        # 2. Vector search
        search_results = self.vector_store.search(q_emb, settings.k)

        # 3. Retrieve full documents
        chunks = [self.doc_store.get_chunk(r.chunk_id) for r in search_results]

        # 4. Rerank and take top_k
        reranked_chunks = self.reranker.rerank(question, chunks)[:settings.top_k]

        # 5. Format prompt
        prompt = get_user_prompt(question, reranked_chunks)

        # 6. Generate answer
        answer = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )

        # 7. Store in cache with document IDs
        chunk_ids = [c.id for c in reranked_chunks]
        self.cache.set(question, answer, chunk_ids)

        return answer, reranked_chunks




