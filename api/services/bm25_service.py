"""BM25 keyword search service."""

import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import jieba
from rank_bm25 import BM25Okapi

from .knowledge_base import load_knowledge_base

_STOPWORDS = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")

_bm25: BM25Okapi | None = None
_tokenized_corpus: list[list[str]] | None = None


def tokenize(text: str) -> list[str]:
    """Chinese tokenization with jieba, filtering stopwords and single chars."""
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in _STOPWORDS]


def _ensure_index():
    global _bm25, _tokenized_corpus
    if _bm25 is not None:
        return
    docs = load_knowledge_base()
    _tokenized_corpus = [tokenize(d["title"] + " " + d["content"]) for d in docs]
    _bm25 = BM25Okapi(_tokenized_corpus)


def search(query: str, top_k: int = 3) -> dict:
    _ensure_index()
    docs = load_knowledge_base()

    t0 = time.perf_counter()
    query_tokens = tokenize(query)
    scores = _bm25.get_scores(query_tokens)

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scored[:top_k]:
        doc = docs[idx]
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": round(float(score), 4),
            "tokens": tokenize(doc["title"] + " " + doc["content"])[:20],
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "results": results,
        "query_tokens": query_tokens,
        "elapsed_ms": elapsed_ms,
    }
