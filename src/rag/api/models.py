from typing import List

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: dict[str, str]

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model: str

class QueryRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    doc_id: str
    text: str
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]