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