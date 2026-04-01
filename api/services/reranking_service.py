"""Reranking service with cross-encoder."""

import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from .knowledge_base import load_knowledge_base

_STOPWORDS = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")

_reranker = None
_model = None
_doc_embeddings = None
_bm25: BM25Okapi | None = None

RERANKER_MODEL = "BAAI/bge-reranker-base"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in _STOPWORDS]


def _ensure_reranker():
    global _reranker
    if _reranker is not None:
        return
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder(RERANKER_MODEL)


def _ensure_embedding():
    global _model, _doc_embeddings
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(EMBEDDING_MODEL)
    docs = load_knowledge_base()
    texts = [d["title"] + " " + d["content"] for d in docs]
    _doc_embeddings = _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def _ensure_bm25():
    global _bm25
    if _bm25 is not None:
        return
    docs = load_knowledge_base()
    corpus = [_tokenize(d["title"] + " " + d["content"]) for d in docs]
    _bm25 = BM25Okapi(corpus)


def _rrf_fusion(
    bm25_results: list[tuple[int, float]],
    embed_results: list[tuple[int, float]],
    k: int = 60,
    bm25_weight: float = 0.5,
    embed_weight: float = 0.5,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (doc_idx, _) in enumerate(bm25_results, 1):
        scores[doc_idx] = scores.get(doc_idx, 0) + bm25_weight / (k + rank)
    for rank, (doc_idx, _) in enumerate(embed_results, 1):
        scores[doc_idx] = scores.get(doc_idx, 0) + embed_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rerank_search(query: str, top_k: int = 5) -> dict:
    """Execute hybrid search + cross-encoder reranking."""
    _ensure_embedding()
    _ensure_bm25()
    _ensure_reranker()
    docs = load_knowledge_base()

    t0 = time.perf_counter()

    # Stage 1: BM25 retrieval
    bm25_scores = _bm25.get_scores(_tokenize(query))
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:10]

    # Stage 1: Embedding retrieval
    q_vec = _model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ _doc_embeddings.T).flatten()
    embed_ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)[:10]

    # RRF fusion
    hybrid_ranked = _rrf_fusion(bm25_ranked, embed_ranked)

    # Stage 2: Cross-encoder reranking on top candidates
    candidate_count = min(len(docs), max(top_k, 10))
    candidate_indices = [idx for idx, _ in hybrid_ranked[:candidate_count]]
    candidate_docs = [docs[idx] for idx in candidate_indices]

    pairs = [(query, d["title"] + " " + d["content"]) for d in candidate_docs]
    ce_scores = _reranker.predict(pairs).tolist()

    # Build before-rerank results
    before_results = []
    for rank, (idx, score) in enumerate(hybrid_ranked[:top_k], 1):
        doc = docs[idx]
        before_results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": round(float(score), 6),
            "rank": rank,
        })

    # Build after-rerank results (sorted by CE scores)
    scored = list(zip(candidate_indices, ce_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build rank maps for change calculation
    before_rank_map = {idx: rank for rank, (idx, _) in enumerate(hybrid_ranked[:candidate_count], 1)}

    after_results = []
    for new_rank, (idx, ce_score) in enumerate(scored[:top_k], 1):
        doc = docs[idx]
        original_rank = before_rank_map.get(idx, candidate_count + 1)
        rank_change = original_rank - new_rank
        after_results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": round(float(ce_score), 4),
            "original_rank": original_rank,
            "new_rank": new_rank,
            "rank_change": rank_change,
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "before_rerank": before_results,
        "after_rerank": after_results,
        "elapsed_ms": elapsed_ms,
        "reranker_model": RERANKER_MODEL,
    }
