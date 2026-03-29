"""FastAPI application entry point for rag-exploration API."""

import os
import warnings

# --- Environment setup (MUST be before any library imports) ---
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import bm25, embedding, rag
from .services import knowledge_base, embedding_service, rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load knowledge base and embedding model
    knowledge_base.load_knowledge_base()
    embedding_service._ensure_model()
    yield


app = FastAPI(
    title="RAG Exploration API",
    description="Backend API for RAG technique exploration and visualization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bm25.router, prefix="/api")
app.include_router(embedding.router, prefix="/api")
app.include_router(rag.router, prefix="/api")


@app.get("/api/health")
def health_check():
    ollama_status = rag_service.check_ollama()
    return {
        "status": "ok",
        "documents_loaded": knowledge_base.get_doc_count(),
        "embedding_model": embedding_service.DEFAULT_MODEL,
        "embedding_dim": embedding_service.get_embedding_dim(),
        "ollama": ollama_status,
    }
