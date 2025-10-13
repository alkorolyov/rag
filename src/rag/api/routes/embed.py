from fastapi import APIRouter, Depends
from fastapi.requests import Request
from fastapi.responses import Response

from rag.api.dependencies import get_embedder
from rag.api.models import EmbedRequest, EmbedResponse
from rag.embeddings import BaseEmbedder

router = APIRouter()

@router.post('/embed')
async def embed(
        request: EmbedRequest,
        embedder: BaseEmbedder = Depends(get_embedder),
):

    return EmbedResponse(
        embedding=embedder.embed_text(request.text).tolist(),
        dimension=embedder.dimension,
        model=embedder.model_name
    )