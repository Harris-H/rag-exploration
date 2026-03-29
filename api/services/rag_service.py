"""RAG query service using Ollama LLM."""

import json
import os
import time
from typing import AsyncGenerator

import httpx

from .embedding_service import _ensure_model, _model, _doc_embeddings
from .knowledge_base import load_knowledge_base
import numpy as np

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = (
    "你是一个专业的 RAG 知识库助手，基于提供的参考文档来回答用户问题。"
    "回答要准确、简洁，不要编造参考文档中没有的信息。"
)


def _build_rag_prompt(query: str, context_docs: list[tuple[dict, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(context_docs, 1):
        parts.append(f"[参考文档{i}] {doc['title']}\n{doc['content']}")
    context_text = "\n\n".join(parts)
    return (
        f"请根据以下参考文档回答用户的问题。如果参考文档中没有相关信息，请如实说明。\n\n"
        f"---参考文档---\n{context_text}\n---参考文档结束---\n\n"
        f"用户问题：{query}\n\n"
        f"请用中文回答，条理清晰，适当引用参考文档中的内容。"
    )


def _retrieve(query: str, top_k: int = 3) -> list[tuple[dict, float]]:
    _ensure_model()
    docs = load_knowledge_base()
    q_vec = _model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ _doc_embeddings.T).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(docs[i], float(sims[i])) for i in top_indices]


def _get_http_client() -> httpx.Client:
    return httpx.Client(proxy=None, timeout=120)


def _get_async_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(proxy=None, timeout=120)


def query(query_text: str, top_k: int = 3) -> dict:
    """Synchronous RAG query returning full answer."""
    t0 = time.perf_counter()

    retrieval_results = _retrieve(query_text, top_k)
    prompt = _build_rag_prompt(query_text, retrieval_results)

    client = _get_http_client()
    try:
        # RAG answer
        rag_resp = client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        rag_resp.raise_for_status()
        rag_answer = rag_resp.json().get("message", {}).get("content", "")

        # Direct answer (no RAG)
        direct_resp = client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个AI助手，请简洁回答问题。"},
                    {"role": "user", "content": f"请用中文简洁回答：{query_text}"},
                ],
                "stream": False,
            },
        )
        direct_resp.raise_for_status()
        direct_answer = direct_resp.json().get("message", {}).get("content", "")
    finally:
        client.close()

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    formatted_retrieval = [
        {"title": doc["title"], "content": doc["content"], "score": round(score, 4)}
        for doc, score in retrieval_results
    ]

    return {
        "rag_answer": rag_answer,
        "direct_answer": direct_answer,
        "retrieval_results": formatted_retrieval,
        "prompt": prompt,
        "elapsed_ms": elapsed_ms,
    }


async def query_stream(query_text: str, top_k: int = 3) -> AsyncGenerator[dict, None]:
    """Async generator yielding SSE events for streaming RAG."""
    # Step 1: Retrieval
    retrieval_results = _retrieve(query_text, top_k)
    formatted_retrieval = [
        {"title": doc["title"], "content": doc["content"], "score": round(score, 4)}
        for doc, score in retrieval_results
    ]
    yield {"event": "retrieval", "data": json.dumps(formatted_retrieval, ensure_ascii=False)}

    # Step 2: Build prompt
    prompt = _build_rag_prompt(query_text, retrieval_results)
    yield {"event": "prompt", "data": prompt}

    # Step 3: Stream tokens from Ollama
    collected_tokens: list[str] = []
    async with _get_async_http_client() as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    collected_tokens.append(token)
                    yield {"event": "token", "data": token}

    # Step 4: Done
    full_answer = "".join(collected_tokens)
    yield {
        "event": "done",
        "data": json.dumps({
            "full_answer": full_answer,
            "retrieval_count": len(formatted_retrieval),
            "model": LLM_MODEL,
        }, ensure_ascii=False),
    }


def check_ollama() -> dict:
    """Check Ollama connectivity and available models."""
    try:
        client = _get_http_client()
        try:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"status": "connected", "models": models}
        finally:
            client.close()
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}
