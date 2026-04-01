"""Chunking strategies service."""

import re
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
from .knowledge_base import load_knowledge_base

_model = None
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def _ensure_model():
    global _model
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(DEFAULT_MODEL)


# ============================================================
# 策略一：固定长度分块
# ============================================================
def chunk_fixed(text: str, chunk_size: int = 150) -> list[str]:
    """按固定字符数切分，不考虑语义边界。"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ============================================================
# 策略二：句子边界分块
# ============================================================
def chunk_sentence(text: str, max_chunk_size: int = 150) -> list[str]:
    """在句子边界处切分，将句子累积到目标大小。"""
    sentences = re.split(r"(?<=[。！？；\n])", text)
    sentences = [s for s in sentences if s.strip()]

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chunk_size and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ============================================================
# 策略三：滑动窗口分块
# ============================================================
def chunk_sliding(text: str, chunk_size: int = 150, overlap: int = 50) -> list[str]:
    """固定窗口大小 + 重叠区域，保留边界上下文。"""
    if overlap >= chunk_size:
        overlap = chunk_size // 3
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


# ============================================================
# 策略四：递归分块
# ============================================================
def chunk_recursive(text: str, max_chunk_size: int = 150) -> list[str]:
    """多级递归分块：段落 → 句子 → 逗号 → 固定切分。"""
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []

    separators = ["\n\n", "。", "！", "？", "；", "，", "、"]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                piece = part + sep if sep not in ("\n\n",) else part
                if len(current) + len(piece) > max_chunk_size and current:
                    chunks.extend(chunk_recursive(current.strip(), max_chunk_size))
                    current = piece
                else:
                    current += piece
            if current.strip():
                chunks.extend(chunk_recursive(current.strip(), max_chunk_size))
            if chunks:
                return chunks

    return chunk_fixed(text, max_chunk_size)


# ============================================================
# 策略对比接口
# ============================================================
def compare_strategies(query: str, chunk_size: int = 150, overlap: int = 50) -> dict:
    """Compare all 4 chunking strategies for a query."""
    _ensure_model()
    docs = load_knowledge_base()

    # 拼接所有文档作为源文本
    full_text = "\n\n".join(d["title"] + "。" + d["content"] for d in docs)

    t0 = time.perf_counter()

    strategies = {
        "fixed": chunk_fixed(full_text, chunk_size),
        "sentence": chunk_sentence(full_text, chunk_size),
        "sliding": chunk_sliding(full_text, chunk_size, overlap),
        "recursive": chunk_recursive(full_text, chunk_size),
    }

    # 编码查询
    q_vec = _model.encode([query], normalize_embeddings=True)

    results = {}
    for name, chunks in strategies.items():
        if not chunks:
            results[name] = {
                "strategy_name": name,
                "num_chunks": 0,
                "avg_chunk_len": 0,
                "min_chunk_len": 0,
                "max_chunk_len": 0,
                "top_chunks": [],
            }
            continue

        # 编码分块
        chunk_vecs = _model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False
        )
        sims = (q_vec @ chunk_vecs.T).flatten()
        top_indices = np.argsort(sims)[::-1][:3]

        lengths = [len(c) for c in chunks]
        results[name] = {
            "strategy_name": name,
            "num_chunks": len(chunks),
            "avg_chunk_len": round(sum(lengths) / len(lengths), 1),
            "min_chunk_len": min(lengths),
            "max_chunk_len": max(lengths),
            "top_chunks": [
                {
                    "text": chunks[i][:200],
                    "score": round(float(sims[i]), 4),
                    "full_length": len(chunks[i]),
                }
                for i in top_indices
            ],
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "strategies": results,
        "query": query,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "source_text_length": len(full_text),
        "elapsed_ms": elapsed_ms,
    }
