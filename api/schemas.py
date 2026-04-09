from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    question:  str = Field(..., min_length=3, max_length=1000)
    top_k:     int = Field(default=5, ge=1, le=20)

    model_config = {"json_schema_extra": {"example": {"question": "What is the refund policy?", "top_k": 5}}}


class SourceDocument(BaseModel):
    source:      str
    chunk_index: int
    score:       float
    preview:     str          # first 200 chars of the chunk


class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  List[SourceDocument]


class HealthResponse(BaseModel):
    status:       str         # "ok" | "degraded"
    ollama:       bool
    vector_store: bool
    total_chunks: int
