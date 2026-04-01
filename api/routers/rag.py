"""RAG query router with SSE streaming support."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..services import rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get("/models")
async def list_models():
    """List available Ollama LLM models."""
    info = rag_service.check_ollama()
    if info["status"] != "connected":
        return {"models": [], "default": rag_service.LLM_MODEL}
    return {"models": info["models"], "default": rag_service.LLM_MODEL}


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)
    stream: bool = False


@router.post("/query")
async def rag_query(req: RAGQueryRequest):
    if not req.stream:
        return rag_service.query(req.query, req.top_k)

    async def event_generator():
        async for event in rag_service.query_stream(req.query, req.top_k):
            yield {"event": event["event"], "data": event["data"]}

    return EventSourceResponse(event_generator())
