from rag.config import Settings
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

    def query(self, question: str, k: int = 100, top_k: int = 5) -> tuple[str, list]:
        """
        Query the RAG pipeline.

        Args:
            question: User's question
            k: Number of initial results from vector search
            top_k: Number of results to keep after reranking

        Returns:
            Tuple of (answer, source_documents)
        """
        # 1. Embed query
        q_emb = self.embedder.embed_text(question)

        # 2. Vector search
        search_results = self.vector_store.search(q_emb, k)

        # 3. Retrieve full documents
        initial_docs = [self.doc_store.get_by_id(r.chunk_id) for r in search_results]

        # 4. Rerank and take top_k
        reranked_docs = self.reranker.rerank(question, initial_docs)[:top_k]

        # 5. Format prompt
        prompt = get_user_prompt(question, reranked_docs)

        # 6. Generate answer
        answer = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )

        return answer, reranked_docs




