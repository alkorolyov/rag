"""Query endpoint for RAG API."""

from fastapi import APIRouter, Depends

from rag.api.dependencies import get_retriever
from rag.api.models import QueryRequest, QueryResponse, SourceDocument
from rag.retrieval import HybridRetriever

router = APIRouter()


@router.post("/query")
async def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> QueryResponse:
    """Query the RAG system.

    Args:
        request: QueryRequest with question

    Returns:
        QueryResponse with retrieved sources (LLM generation can be added)
    """
    results = retriever.search(request.question, k=10)

    sources = [
        SourceDocument(
            doc_id=str(doc.metadata.get("doc_id", "")),
            text=doc.page_content[:500],
            score=doc.metadata.get("score", 0.0),
        )
        for doc in results
    ]

    # Note: LLM generation can be added here via LangChain
    answer = f"Retrieved {len(sources)} relevant documents. LLM generation not yet implemented."

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
    )
