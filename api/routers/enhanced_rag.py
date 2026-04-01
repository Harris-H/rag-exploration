"""Enhanced RAG router with configurable pipeline options."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..services import enhanced_rag_service

router = APIRouter(prefix="/rag", tags=["enhanced-rag"])


class EnhancedRAGRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)
    use_expansion: bool = False
    use_chunking: bool = True
    chunk_strategy: str = Field(default="recursive", pattern="^(fixed|sentence|sliding|recursive)$")
    chunk_size: int = Field(default=200, ge=50, le=500)
    use_hybrid: bool = True
    use_reranking: bool = True


@router.post("/enhanced-query")
async def enhanced_query(req: EnhancedRAGRequest):
    async def event_generator():
        async for event in enhanced_rag_service.enhanced_query_stream(
            query=req.query,
            top_k=req.top_k,
            use_expansion=req.use_expansion,
            use_chunking=req.use_chunking,
            chunk_strategy=req.chunk_strategy,
            chunk_size=req.chunk_size,
            use_hybrid=req.use_hybrid,
            use_reranking=req.use_reranking,
        ):
            yield {"event": event["event"], "data": event["data"]}

    return EventSourceResponse(event_generator())
