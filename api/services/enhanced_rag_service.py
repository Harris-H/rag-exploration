"""Enhanced RAG service with query expansion, chunking, hybrid retrieval, and reranking.

Implements a configurable 7-step pipeline:
  User Input → Query Expansion → Chunking → Hybrid Retrieval → Reranking → Prompt → LLM Streaming
"""

import json
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator

import numpy as np
from rank_bm25 import BM25Okapi

from . import embedding_service, reranking_service, rag_service
from .chunking_service import chunk_fixed, chunk_sentence, chunk_sliding, chunk_recursive
from .knowledge_base import load_knowledge_base

_STOPWORDS = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")

CHUNK_STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "sliding": chunk_sliding,
    "recursive": chunk_recursive,
}


@dataclass
class ChunkInfo:
    """A single chunk with source document metadata."""
    text: str
    doc_id: str
    doc_title: str
    chunk_idx: int


@dataclass
class ChunkIndex:
    """Pre-built index for chunked document retrieval."""
    chunks: list[ChunkInfo]
    embeddings: np.ndarray
    bm25: BM25Okapi
    strategy: str
    chunk_size: int
    overlap: int


# (strategy, chunk_size, overlap) → ChunkIndex
_chunk_index_cache: dict[tuple[str, int, int], ChunkIndex] = {}


def _tokenize(text: str) -> list[str]:
    import jieba
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in _STOPWORDS]


def _build_chunk_index(
    strategy: str = "recursive",
    chunk_size: int = 200,
    overlap: int = 50,
) -> ChunkIndex:
    """Build or retrieve cached chunk index for the knowledge base."""
    cache_key = (strategy, chunk_size, overlap)
    if cache_key in _chunk_index_cache:
        return _chunk_index_cache[cache_key]

    embedding_service._ensure_model()
    docs = load_knowledge_base()

    all_chunks: list[ChunkInfo] = []
    for doc in docs:
        full_text = doc["title"] + "。" + doc["content"]
        doc_id = doc.get("id", f"doc_{docs.index(doc) + 1:02d}")

        if strategy == "sliding":
            raw_chunks = chunk_sliding(full_text, chunk_size, overlap)
        elif strategy == "sentence":
            raw_chunks = chunk_sentence(full_text, chunk_size)
        elif strategy == "fixed":
            raw_chunks = chunk_fixed(full_text, chunk_size)
        else:
            raw_chunks = chunk_recursive(full_text, chunk_size)

        for idx, text in enumerate(raw_chunks):
            all_chunks.append(ChunkInfo(
                text=text,
                doc_id=doc_id,
                doc_title=doc["title"],
                chunk_idx=idx,
            ))

    chunk_texts = [c.text for c in all_chunks]
    chunk_embeddings = embedding_service._model.encode(
        chunk_texts, normalize_embeddings=True, show_progress_bar=False,
    )

    tokenized = [_tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized)

    index = ChunkIndex(
        chunks=all_chunks,
        embeddings=chunk_embeddings,
        bm25=bm25,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    _chunk_index_cache[cache_key] = index
    return index


def _rrf_fusion(
    bm25_results: list[tuple[int, float]],
    embed_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_results, 1):
        scores[idx] = scores.get(idx, 0) + 0.5 / (k + rank)
    for rank, (idx, _) in enumerate(embed_results, 1):
        scores[idx] = scores.get(idx, 0) + 0.5 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _multi_query_rrf(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """RRF fusion across multiple ranked lists (one per query variant)."""
    scores: dict[int, float] = {}
    weight = 1.0 / len(ranked_lists)
    for ranked in ranked_lists:
        for rank, (idx, score) in enumerate(ranked, 1):
            scores[idx] = scores.get(idx, 0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _expand_query(query: str, num_variants: int = 3, model: str | None = None) -> list[str]:
    """Use LLM to generate diverse query variants from different angles."""
    import httpx

    llm = model or rag_service.LLM_MODEL
    prompt = (
        f"你是一个信息检索专家。用户提出了一个问题，你需要从不同的角度生成 {num_variants} 个检索查询，"
        f"帮助从知识库中找到更多相关文档。\n\n"
        f"要求：\n"
        f"- 每个查询必须使用不同的关键词和切入角度\n"
        f"- 不要简单换句式，要从不同技术方向切入（如：换专业术语、换子问题、换具体方法）\n"
        f"- 查询要简短精炼（10-25字），像搜索引擎关键词\n"
        f"- 直接输出查询内容，每行一个，不要编号，不要加\"用户问题\"等前缀\n\n"
        f"示例：\n"
        f"问题：怎么让模型回答更准确？\n"
        f"RAG 检索增强生成 减少幻觉\n"
        f"提示工程 Prompt 模板优化技巧\n"
        f"Cross-Encoder 重排序提升检索质量\n\n"
        f"问题：{query}\n"
    )

    try:
        with httpx.Client(proxy=None, timeout=30) as client:
            resp = client.post(
                f"{rag_service.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": llm,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                },
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")

        variants = []
        for line in content.strip().split("\n"):
            line = line.strip()
            line = re.sub(r"^[\d]+[.、)）]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = re.sub(r"^(用户)?问题[：:]\s*", "", line)
            line = re.sub(r"^查询\d*[：:]\s*", "", line)
            line = line.strip()
            if line and line != query and len(line) > 3:
                variants.append(line)
        return variants[:num_variants]
    except Exception:
        return []


def _retrieve_chunks(
    query: str,
    index: ChunkIndex,
    top_k: int = 5,
    use_hybrid: bool = True,
) -> list[tuple[ChunkInfo, float]]:
    """Retrieve most relevant chunks."""
    embedding_service._ensure_model()
    q_vec = embedding_service._model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ index.embeddings.T).flatten()

    if not use_hybrid:
        embed_ranked = np.argsort(sims)[::-1][:top_k]
        return [(index.chunks[i], float(sims[i])) for i in embed_ranked]

    embed_ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)[:top_k * 2]
    bm25_scores = index.bm25.get_scores(_tokenize(query))
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k * 2]

    hybrid_ranked = _rrf_fusion(bm25_ranked, embed_ranked)
    return [(index.chunks[idx], float(score)) for idx, score in hybrid_ranked[:top_k]]


def _retrieve_chunks_multi_query(
    queries: list[str],
    index: ChunkIndex,
    top_k: int = 5,
    use_hybrid: bool = True,
) -> list[tuple[ChunkInfo, float]]:
    """Retrieve chunks using multiple query variants, fused via RRF."""
    embedding_service._ensure_model()

    per_query_ranked: list[list[tuple[int, float]]] = []
    for q in queries:
        q_vec = embedding_service._model.encode([q], normalize_embeddings=True)
        sims = (q_vec @ index.embeddings.T).flatten()

        if not use_hybrid:
            ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)
            per_query_ranked.append(ranked[:top_k * 3])
        else:
            embed_ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)[:top_k * 3]
            bm25_scores = index.bm25.get_scores(_tokenize(q))
            bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k * 3]
            hybrid_ranked = _rrf_fusion(bm25_ranked, embed_ranked)
            per_query_ranked.append(hybrid_ranked[:top_k * 3])

    fused = _multi_query_rrf(per_query_ranked)
    return [(index.chunks[idx], float(score)) for idx, score in fused[:top_k]]


def _rerank_chunks(
    query: str,
    candidates: list[tuple[ChunkInfo, float]],
    top_k: int = 3,
) -> dict:
    """Rerank chunk candidates using Cross-Encoder. Returns before/after comparison."""
    reranking_service._ensure_reranker()

    pairs = [(query, c.text) for c, _ in candidates]
    ce_scores = reranking_service._reranker.predict(pairs).tolist()

    before = []
    for rank, (chunk, score) in enumerate(candidates, 1):
        before.append({
            "text": chunk.text[:200],
            "doc_title": chunk.doc_title,
            "doc_id": chunk.doc_id,
            "score": round(float(score), 4),
            "rank": rank,
        })

    scored = list(zip(range(len(candidates)), ce_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    after = []
    reranked = []
    for new_rank, (orig_idx, ce_score) in enumerate(scored[:top_k], 1):
        chunk, _ = candidates[orig_idx]
        after.append({
            "text": chunk.text[:200],
            "doc_title": chunk.doc_title,
            "doc_id": chunk.doc_id,
            "score": round(float(ce_score), 4),
            "original_rank": orig_idx + 1,
            "new_rank": new_rank,
            "rank_change": (orig_idx + 1) - new_rank,
        })
        reranked.append((chunk, float(ce_score)))

    return {"before": before[:top_k], "after": after, "candidates": reranked}


def _build_enhanced_prompt(query: str, chunks: list[tuple[ChunkInfo, float]]) -> str:
    """Build RAG prompt from retrieved chunks with citation instructions."""
    parts = []
    for i, (chunk, _) in enumerate(chunks, 1):
        parts.append(f"[参考片段{i}] (来源: {chunk.doc_title})\n{chunk.text}")
    context_text = "\n\n".join(parts)
    return (
        f"请根据以下参考资料回答用户的问题。如果参考资料中没有相关信息，请如实说明。\n\n"
        f"---参考资料---\n{context_text}\n---参考资料结束---\n\n"
        f"用户问题：{query}\n\n"
        f"回答要求：\n"
        f"- 用中文回答，条理清晰\n"
        f"- 在引用参考资料内容时，在相关句子末尾标注来源编号，格式为[1]、[2]、[3]\n"
        f"- 编号对应上面的参考片段编号，可以综合标注如[1][3]"
    )


def _post_process_citations(
    answer: str,
    chunks: list[tuple[ChunkInfo, float]],
) -> str:
    """Ensure answer has proper inline citations via text-overlap matching.

    Strategy:
    1. If the LLM already produced valid [N] citations, keep them.
    2. Otherwise, split answer into sentences, compute overlap with each
       source chunk, and inject [N] at the end of sentences with high overlap.
    """
    # Clean up malformed brackets the LLM might produce
    answer = re.sub(r"\[\s*\]", "", answer)       # empty []
    answer = re.sub(r"\[R(\d+)\]", r"[\1]", answer)  # [R3] → [3]

    # Check if valid citations already exist
    existing = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
    valid_indices = set(range(1, len(chunks) + 1))
    if existing & valid_indices:
        return answer.strip()

    # No valid citations found → inject via text overlap
    def _overlap_score(sentence: str, chunk_text: str) -> float:
        s_chars = set(sentence)
        c_chars = set(chunk_text)
        if not s_chars:
            return 0.0
        return len(s_chars & c_chars) / len(s_chars | c_chars)

    # Split answer into sentences (Chinese punctuation aware)
    sentences = re.split(r"(?<=[。！？\n])", answer)
    result_parts: list[str] = []

    for sent in sentences:
        stripped = sent.strip()
        if not stripped or len(stripped) < 5:
            result_parts.append(sent)
            continue

        # Find best matching chunk
        best_idx, best_score = 0, 0.0
        for i, (chunk, _) in enumerate(chunks, 1):
            score = _overlap_score(stripped, chunk.text)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score > 0.15 and best_idx > 0:
            # Insert citation before the sentence-ending punctuation
            m = re.search(r"[。！？]$", sent)
            if m:
                pos = m.start()
                sent = sent[:pos] + f"[{best_idx}]" + sent[pos:]
            else:
                sent = sent.rstrip() + f"[{best_idx}]"

        result_parts.append(sent)

    return "".join(result_parts).strip()


async def enhanced_query_stream(
    query: str,
    top_k: int = 3,
    use_chunking: bool = True,
    chunk_strategy: str = "recursive",
    chunk_size: int = 200,
    use_hybrid: bool = True,
    use_reranking: bool = True,
    use_expansion: bool = False,
    model: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Enhanced RAG pipeline streaming SSE events."""
    t0 = time.perf_counter()
    llm_model = model or rag_service.LLM_MODEL

    config = {
        "use_expansion": use_expansion,
        "use_chunking": use_chunking,
        "chunk_strategy": chunk_strategy if use_chunking else None,
        "chunk_size": chunk_size if use_chunking else None,
        "use_hybrid": use_hybrid,
        "use_reranking": use_reranking,
        "top_k": top_k,
        "model": llm_model,
    }
    yield {"event": "config", "data": json.dumps(config, ensure_ascii=False)}

    # Track final chunks for citation post-processing
    citation_chunks: list[tuple[ChunkInfo, float]] = []

    # Step 1 (optional): Query Expansion
    all_queries = [query]
    if use_expansion:
        variants = _expand_query(query, model=llm_model)
        if variants:
            all_queries = [query] + variants
        yield {
            "event": "expansion",
            "data": json.dumps({
                "original": query,
                "variants": variants,
                "total_queries": len(all_queries),
            }, ensure_ascii=False),
        }

    if use_chunking:
        # Step 2: Chunking
        index = _build_chunk_index(chunk_strategy, chunk_size)
        chunking_info = {
            "strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "num_chunks": len(index.chunks),
            "num_source_docs": len(load_knowledge_base()),
        }
        yield {"event": "chunking", "data": json.dumps(chunking_info, ensure_ascii=False)}

        # Step 3: Retrieve chunks (multi-query if expanded)
        retrieve_count = top_k * 3 if use_reranking else top_k
        if len(all_queries) > 1:
            candidates = _retrieve_chunks_multi_query(
                all_queries, index, retrieve_count, use_hybrid,
            )
        else:
            candidates = _retrieve_chunks(query, index, retrieve_count, use_hybrid)

        retrieval_data = [
            {
                "text": c.text[:200],
                "doc_title": c.doc_title,
                "doc_id": c.doc_id,
                "score": round(float(score), 4),
            }
            for c, score in candidates
        ]
        yield {"event": "retrieval", "data": json.dumps(retrieval_data, ensure_ascii=False)}

        # Step 4: Rerank (optional)
        if use_reranking and len(candidates) > 0:
            rerank_result = _rerank_chunks(query, candidates, top_k)
            yield {"event": "reranking", "data": json.dumps({
                "before": rerank_result["before"],
                "after": rerank_result["after"],
            }, ensure_ascii=False)}
            final_chunks = rerank_result["candidates"]
        else:
            final_chunks = candidates[:top_k]

        # Step 5: Build prompt from chunks
        prompt = _build_enhanced_prompt(query, final_chunks)
        citation_chunks = final_chunks
        sources = [
            {"index": i + 1, "doc_title": c.doc_title, "doc_id": c.doc_id}
            for i, (c, _) in enumerate(final_chunks)
        ]
    else:
        # Fallback: use original full-doc retrieval
        retrieval_results = rag_service._retrieve(query, top_k)
        retrieval_data = [
            {"title": doc["title"], "content": doc["content"], "score": round(score, 4)}
            for doc, score in retrieval_results
        ]
        yield {"event": "retrieval", "data": json.dumps(retrieval_data, ensure_ascii=False)}
        prompt = rag_service._build_rag_prompt(query, retrieval_results)
        sources = [
            {"index": i + 1, "doc_title": doc["title"], "doc_id": doc.get("id", "")}
            for i, (doc, _) in enumerate(retrieval_results)
        ]

    yield {"event": "prompt", "data": prompt}

    # Step 6: Stream LLM tokens
    collected_tokens: list[str] = []
    async with rag_service._get_async_http_client() as client:
        async with client.stream(
            "POST",
            f"{rag_service.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": rag_service.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "think": False,
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

    # Step 7: Done — post-process citations and strip thinking tags
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    full_answer = "".join(collected_tokens)
    # Strip any leaked <think>...</think> blocks
    full_answer = re.sub(r"<think>.*?</think>", "", full_answer, flags=re.DOTALL).strip()
    if citation_chunks:
        full_answer = _post_process_citations(full_answer, citation_chunks)
    yield {
        "event": "done",
        "data": json.dumps({
            "full_answer": full_answer,
            "elapsed_ms": elapsed_ms,
            "model": llm_model,
            "config": config,
            "sources": sources,
        }, ensure_ascii=False),
    }
