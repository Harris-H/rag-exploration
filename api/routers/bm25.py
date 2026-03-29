"""BM25 search router."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services import bm25_service

router = APIRouter(prefix="/bm25", tags=["BM25"])


class BM25SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/search")
def bm25_search(req: BM25SearchRequest):
    return bm25_service.search(req.query, req.top_k)
