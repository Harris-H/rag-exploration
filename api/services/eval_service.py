"""Evaluation metrics service."""

import math
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

# ── 评估查询集（人工标注 ground truth）──────────────────────────────────────

EVAL_QUERIES = [
    {
        "query": "什么是RAG技术？它有什么作用？",
        "relevant_docs": ["doc_01", "doc_04"],
        "description": "RAG 概念理解",
    },
    {
        "query": "BM25和TF-IDF有什么区别？",
        "relevant_docs": ["doc_02", "doc_06"],
        "description": "经典检索算法对比",
    },
    {
        "query": "如何将文本转换为向量进行语义搜索？",
        "relevant_docs": ["doc_07", "doc_03"],
        "description": "语义检索技术",
    },
    {
        "query": "怎么在本地电脑上运行大模型？",
        "relevant_docs": ["doc_10", "doc_04"],
        "description": "本地部署",
    },
    {
        "query": "文档应该怎么切分？分块大小如何选择？",
        "relevant_docs": ["doc_05"],
        "description": "分块策略",
    },
    {
        "query": "混合检索和纯向量检索哪个好？",
        "relevant_docs": ["doc_09", "doc_07", "doc_02"],
        "description": "检索方法选择",
    },
    {
        "query": "知识图谱在RAG中有什么用？",
        "relevant_docs": ["doc_08", "doc_01"],
        "description": "图增强RAG",
    },
    {
        "query": "向量数据库有哪些选择？",
        "relevant_docs": ["doc_03"],
        "description": "向量存储",
    },
    {
        "query": "怎么让AI回答问题时不胡说八道？",
        "relevant_docs": ["doc_01", "doc_12"],
        "description": "RAG减少幻觉（自然语言风格查询）",
    },
    {
        "query": "搜索引擎的排序和RAG的排序有什么不同？",
        "relevant_docs": ["doc_11", "doc_14", "doc_09"],
        "description": "搜索引擎 vs RAG 排序（干扰项区分）",
    },
]

# ── 分词 & 索引 ──────────────────────────────────────────────────────────


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


# ── 评估指标 ──────────────────────────────────────────────────────────────


def mean_reciprocal_rank(ranked_doc_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR: 第一个相关文档的排名的倒数"""
    for i, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """P@K: 前K个结果中相关文档的比例"""
    top_k = ranked_doc_ids[:k]
    relevant_count = sum(1 for d in top_k if d in relevant_ids)
    return relevant_count / k


def recall_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """R@K: 前K个结果覆盖了多少比例的相关文档"""
    top_k = ranked_doc_ids[:k]
    relevant_count = sum(1 for d in top_k if d in relevant_ids)
    return relevant_count / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@K: 归一化折扣累积增益"""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], 1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_rels = sorted(
        [1.0] * min(len(relevant_ids), k) + [0.0] * max(0, k - len(relevant_ids)),
        reverse=True,
    )
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# ── 检索函数 ──────────────────────────────────────────────────────────────


def _search_bm25(query: str, docs: list[dict], top_k: int = 10) -> list[str]:
    _ensure_bm25()
    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [docs[idx]["id"] for idx, _ in ranked[:top_k]]


def _search_embedding(query: str, docs: list[dict], top_k: int = 10) -> list[str]:
    _ensure_model()
    q_vec = _model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ _doc_embeddings.T).flatten()
    ranked = np.argsort(sims)[::-1][:top_k]
    return [docs[idx]["id"] for idx in ranked]


def _search_hybrid(query: str, docs: list[dict], top_k: int = 10, rrf_k: int = 60) -> list[str]:
    _ensure_model()
    _ensure_bm25()

    # BM25
    bm25_scores = _bm25.get_scores(_tokenize(query))
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)

    # Embedding
    q_vec = _model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ _doc_embeddings.T).flatten()
    embed_ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)

    # RRF
    rrf_scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_ranked, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 0.5 / (rrf_k + rank)
    for rank, (idx, _) in enumerate(embed_ranked, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 0.5 / (rrf_k + rank)

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[idx]["id"] for idx, _ in sorted_rrf[:top_k]]


# ── 单条查询评估 ─────────────────────────────────────────────────────────


def _evaluate_method(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    return {
        "mrr": round(mean_reciprocal_rank(ranked_ids, relevant_ids), 4),
        "precision_at_k": round(precision_at_k(ranked_ids, relevant_ids, k), 4),
        "recall_at_k": round(recall_at_k(ranked_ids, relevant_ids, k), 4),
        "ndcg_at_k": round(ndcg_at_k(ranked_ids, relevant_ids, k), 4),
    }


# ── 公开 API ──────────────────────────────────────────────────────────────


def evaluate_query(query: str, relevant_doc_ids: list[str], k: int = 3) -> dict:
    """Evaluate a single query across all methods."""
    t0 = time.perf_counter()
    docs = load_knowledge_base()
    relevant = set(relevant_doc_ids)

    bm25_ids = _search_bm25(query, docs)
    emb_ids = _search_embedding(query, docs)
    hyb_ids = _search_hybrid(query, docs)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "query": query,
        "relevant_docs": relevant_doc_ids,
        "methods": {
            "BM25": {
                "ranked_docs": bm25_ids[:k],
                "metrics": _evaluate_method(bm25_ids, relevant, k),
            },
            "Embedding": {
                "ranked_docs": emb_ids[:k],
                "metrics": _evaluate_method(emb_ids, relevant, k),
            },
            "Hybrid": {
                "ranked_docs": hyb_ids[:k],
                "metrics": _evaluate_method(hyb_ids, relevant, k),
            },
        },
        "k": k,
        "elapsed_ms": elapsed_ms,
    }


def evaluate_all(k: int = 3) -> dict:
    """Run full evaluation across all methods and queries."""
    t0 = time.perf_counter()
    docs = load_knowledge_base()
    methods = ["BM25", "Embedding", "Hybrid"]
    metric_names = ["mrr", "precision_at_k", "recall_at_k", "ndcg_at_k"]

    per_query: list[dict] = []

    for eq in EVAL_QUERIES:
        query = eq["query"]
        relevant = set(eq["relevant_docs"])

        bm25_ids = _search_bm25(query, docs)
        emb_ids = _search_embedding(query, docs)
        hyb_ids = _search_hybrid(query, docs)

        per_query.append({
            "query": query,
            "description": eq["description"],
            "relevant_docs": eq["relevant_docs"],
            "methods": {
                "BM25": {
                    "ranked_docs": bm25_ids[:k],
                    "metrics": _evaluate_method(bm25_ids, relevant, k),
                },
                "Embedding": {
                    "ranked_docs": emb_ids[:k],
                    "metrics": _evaluate_method(emb_ids, relevant, k),
                },
                "Hybrid": {
                    "ranked_docs": hyb_ids[:k],
                    "metrics": _evaluate_method(hyb_ids, relevant, k),
                },
            },
        })

    # 计算聚合指标
    aggregate: dict[str, dict[str, float]] = {}
    for method_name in methods:
        aggregate[method_name] = {}
        for mn in metric_names:
            vals = [q["methods"][method_name]["metrics"][mn] for q in per_query]
            aggregate[method_name][mn] = round(sum(vals) / len(vals), 4)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "k": k,
        "num_queries": len(EVAL_QUERIES),
        "elapsed_ms": elapsed_ms,
    }
