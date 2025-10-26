from fastapi import APIRouter, Depends

from rag.api.dependencies import get_rag_pipeline
from rag.api.models import QueryRequest, QueryResponse, SourceDocument
from rag.pipeline import RAGPipeline

router = APIRouter()

@router.post('/query')
async def query(
        request: QueryRequest,
        rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """
    Query the RAG system with a question.

    Args:
        request: QueryRequest with question and optional k/top_k params

    Returns:
        QueryResponse with answer and source documents
    """
    # Get answer and source documents from pipeline
    answer, sources = rag_pipeline.query(
        question=request.question,
    )

    # Format source documents
    source_docs = [
        SourceDocument(
            doc_id=str(doc.id),
            text=doc.text,
            score=doc.score
        )
        for doc in sources
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=source_docs
    )