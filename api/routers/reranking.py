"""Reranking router."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services import reranking_service

router = APIRouter(prefix="/reranking", tags=["Reranking"])


class RerankRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/search")
def rerank_search(req: RerankRequest):
    return reranking_service.rerank_search(req.query, req.top_k)
