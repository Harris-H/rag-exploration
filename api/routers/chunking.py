"""Chunking strategies router."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services import chunking_service

router = APIRouter(prefix="/chunking", tags=["Chunking"])


class ChunkingCompareRequest(BaseModel):
    query: str
    chunk_size: int = Field(default=150, ge=50, le=1000)
    overlap: int = Field(default=50, ge=0, le=200)


@router.post("/compare")
def compare_strategies(req: ChunkingCompareRequest):
    return chunking_service.compare_strategies(req.query, req.chunk_size, req.overlap)
