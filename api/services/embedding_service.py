"""Embedding and hybrid search service."""

import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from .knowledge_base import load_knowledge_base

_STOPWORDS = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")

_model = None
_doc_embeddings = None
_bm25: BM25Okapi | None = None

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in _STOPWORDS]


def _ensure_model():
    global _model, _doc_embeddings
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(DEFAULT_MODEL)
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


def get_embedding_dim() -> int:
    _ensure_model()
    return int(_model.get_sentence_embedding_dimension())


def search(query: str, top_k: int = 3) -> dict:
    _ensure_model()
    docs = load_knowledge_base()

    t0 = time.perf_counter()
    q_vec = _model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ _doc_embeddings.T).flatten()
    ranked = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in ranked:
        doc = docs[idx]
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": round(float(sims[idx]), 4),
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "results": results,
        "embedding_dim": get_embedding_dim(),
        "elapsed_ms": elapsed_ms,
    }


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


def hybrid_search(query: str, top_k: int = 3) -> dict:
    _ensure_model()
    _ensure_bm25()
    docs = load_knowledge_base()

    t0 = time.perf_counter()

    # BM25 results
    bm25_scores = _bm25.get_scores(_tokenize(query))
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:10]

    # Embedding results
    q_vec = _model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ _doc_embeddings.T).flatten()
    embed_ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)[:10]

    # RRF fusion
    hybrid_ranked = _rrf_fusion(bm25_ranked, embed_ranked)

    def _format(ranked, limit):
        out = []
        for idx, score in ranked[:limit]:
            doc = docs[idx]
            out.append({"title": doc["title"], "content": doc["content"], "score": round(float(score), 4)})
        return out

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "bm25_results": _format(bm25_ranked, top_k),
        "embedding_results": _format(embed_ranked, top_k),
        "hybrid_results": _format(hybrid_ranked, top_k),
        "elapsed_ms": elapsed_ms,
    }
