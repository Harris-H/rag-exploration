"""Embedding and hybrid search router."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services import embedding_service

router = APIRouter(prefix="/embedding", tags=["Embedding"])


class EmbeddingSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/search")
def embedding_search(req: EmbeddingSearchRequest):
    return embedding_service.search(req.query, req.top_k)


@router.post("/hybrid")
def hybrid_search(req: EmbeddingSearchRequest):
    return embedding_service.hybrid_search(req.query, req.top_k)
