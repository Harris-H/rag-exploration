"""
📏 检索评估 Demo —— Retrieval Evaluation Metrics

这个脚本展示：
  1. MRR（Mean Reciprocal Rank）平均倒数排名
  2. NDCG@K（Normalized Discounted Cumulative Gain）
  3. Precision@K 和 Recall@K
  4. 三种检索方法（BM25 / Embedding / Hybrid）的定量对比

用法：uv run evaluation/eval_demo.py
"""

import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── 常量 ────────────────────────────────────────────────────────────────────

console = Console()

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "knowledge_base.json")

_STOPWORDS = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")

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

    # Ideal DCG：相关文档全部排在前面
    ideal_rels = sorted(
        [1.0] * min(len(relevant_ids), k) + [0.0] * max(0, k - len(relevant_ids)),
        reverse=True,
    )
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# ── 分词 & 索引 ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in _STOPWORDS]


def _load_docs() -> list[dict]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_bm25(docs: list[dict]) -> BM25Okapi:
    corpus = [_tokenize(d["title"] + " " + d["content"]) for d in docs]
    return BM25Okapi(corpus)


def _build_embeddings(docs: list[dict]):
    """加载 Embedding 模型并编码文档，返回 (model, doc_embeddings)。"""
    from sentence_transformers import SentenceTransformer

    console.print("  ⏳ 加载 Embedding 模型...", style="dim")
    model = SentenceTransformer(DEFAULT_MODEL)
    texts = [d["title"] + " " + d["content"] for d in docs]
    doc_embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return model, doc_embs


# ── 检索函数 ──────────────────────────────────────────────────────────────

def _search_bm25(query: str, bm25: BM25Okapi, docs: list[dict], top_k: int = 10) -> list[str]:
    """BM25 检索，返回排序后的 doc_id 列表。"""
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [docs[idx]["id"] for idx, _ in ranked[:top_k]]


def _search_embedding(query: str, model, doc_embs, docs: list[dict], top_k: int = 10) -> list[str]:
    """Embedding 检索，返回排序后的 doc_id 列表。"""
    q_vec = model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ doc_embs.T).flatten()
    ranked = np.argsort(sims)[::-1][:top_k]
    return [docs[idx]["id"] for idx in ranked]


def _search_hybrid(
    query: str, bm25: BM25Okapi, model, doc_embs, docs: list[dict], top_k: int = 10,
    rrf_k: int = 60,
) -> list[str]:
    """RRF 混合检索，返回排序后的 doc_id 列表。"""
    # BM25 排名
    bm25_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(bm25_tokens)
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)

    # Embedding 排名
    q_vec = model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ doc_embs.T).flatten()
    embed_ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)

    # RRF 融合
    rrf_scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_ranked, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 0.5 / (rrf_k + rank)
    for rank, (idx, _) in enumerate(embed_ranked, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 0.5 / (rrf_k + rank)

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[idx]["id"] for idx, _ in sorted_rrf[:top_k]]


# ── 评估核心 ──────────────────────────────────────────────────────────────

def _evaluate_method(
    ranked_ids: list[str], relevant_ids: set[str], k: int,
) -> dict[str, float]:
    """计算单次查询在单种方法下的所有指标。"""
    return {
        "mrr": mean_reciprocal_rank(ranked_ids, relevant_ids),
        "precision_at_k": precision_at_k(ranked_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(ranked_ids, relevant_ids, k),
        "ndcg_at_k": ndcg_at_k(ranked_ids, relevant_ids, k),
    }


def _score_bar(value: float, width: int = 15) -> str:
    """生成一个简易的文本分数条。"""
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    console.print(
        Panel(
            "[bold cyan]📏 检索质量评估框架[/bold cyan]\n"
            "[dim]Retrieval Evaluation Metrics — MRR / P@K / R@K / NDCG@K[/dim]",
            box=box.ROUNDED,
            expand=False,
        )
    )

    # ── 1. 加载数据 & 构建索引 ──
    console.print("\n[bold]📂 加载知识库...[/bold]")
    docs = _load_docs()
    console.print(f"  ✅ 共 {len(docs)} 篇文档\n")

    console.print("[bold]🔧 构建检索索引...[/bold]")

    t0 = time.perf_counter()
    bm25 = _build_bm25(docs)
    bm25_time = time.perf_counter() - t0
    console.print(f"  ✅ BM25 索引就绪  ({bm25_time*1000:.0f}ms)")

    t0 = time.perf_counter()
    model, doc_embs = _build_embeddings(docs)
    emb_time = time.perf_counter() - t0
    console.print(f"  ✅ Embedding 索引就绪  ({emb_time*1000:.0f}ms)")

    K = 3  # 评估 top-K
    methods = ["BM25", "Embedding", "Hybrid"]

    # ── 2. 逐查询评估 ──
    console.print(f"\n[bold]📊 开始评估（K={K}，{len(EVAL_QUERIES)} 条查询）...[/bold]\n")

    all_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in methods}

    for qi, eq in enumerate(EVAL_QUERIES, 1):
        query = eq["query"]
        relevant = set(eq["relevant_docs"])
        desc = eq["description"]

        bm25_ids = _search_bm25(query, bm25, docs)
        emb_ids = _search_embedding(query, model, doc_embs, docs)
        hyb_ids = _search_hybrid(query, bm25, model, doc_embs, docs)

        bm25_m = _evaluate_method(bm25_ids, relevant, K)
        emb_m = _evaluate_method(emb_ids, relevant, K)
        hyb_m = _evaluate_method(hyb_ids, relevant, K)

        all_metrics["BM25"].append(bm25_m)
        all_metrics["Embedding"].append(emb_m)
        all_metrics["Hybrid"].append(hyb_m)

        # 每条查询的小表格
        tbl = Table(
            title=f"Q{qi}: {desc}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold yellow",
        )
        tbl.add_column("方法", style="cyan", width=12)
        tbl.add_column("Top-3 文档", width=28)
        tbl.add_column("MRR", justify="right", width=7)
        tbl.add_column(f"P@{K}", justify="right", width=7)
        tbl.add_column(f"R@{K}", justify="right", width=7)
        tbl.add_column(f"NDCG@{K}", justify="right", width=8)

        for method_name, m_dict, ranked_ids in [
            ("BM25", bm25_m, bm25_ids),
            ("Embedding", emb_m, emb_ids),
            ("Hybrid", hyb_m, hyb_ids),
        ]:
            top3_str = ", ".join(ranked_ids[:K])
            # 高亮相关文档
            highlighted_parts = []
            for did in ranked_ids[:K]:
                if did in relevant:
                    highlighted_parts.append(f"[green bold]{did}[/green bold]")
                else:
                    highlighted_parts.append(f"[dim]{did}[/dim]")
            top3_display = ", ".join(highlighted_parts)

            tbl.add_row(
                method_name,
                top3_display,
                f"{m_dict['mrr']:.3f}",
                f"{m_dict['precision_at_k']:.3f}",
                f"{m_dict['recall_at_k']:.3f}",
                f"{m_dict['ndcg_at_k']:.3f}",
            )

        console.print(f"  🔍 [bold]{query}[/bold]")
        console.print(f"     📎 相关文档: {', '.join(sorted(relevant))}")
        console.print(tbl)
        console.print()

    # ── 3. 汇总表 ──
    console.print(
        Panel(
            "[bold cyan]📈 总体评估结果（所有查询平均）[/bold cyan]",
            box=box.ROUNDED,
            expand=False,
        )
    )

    metric_names = ["mrr", "precision_at_k", "recall_at_k", "ndcg_at_k"]
    display_names = {"mrr": "MRR", "precision_at_k": f"P@{K}", "recall_at_k": f"R@{K}", "ndcg_at_k": f"NDCG@{K}"}

    # 计算每种方法的平均指标
    agg: dict[str, dict[str, float]] = {}
    for method_name in methods:
        agg[method_name] = {}
        for mn in metric_names:
            vals = [m[mn] for m in all_metrics[method_name]]
            agg[method_name][mn] = sum(vals) / len(vals)

    summary_tbl = Table(
        title="🏆 方法对比总结",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        title_style="bold",
    )
    summary_tbl.add_column("指标", style="cyan", width=10)
    for method_name in methods:
        summary_tbl.add_column(method_name, justify="center", width=18)

    for mn in metric_names:
        values = {m: agg[m][mn] for m in methods}
        best_method = max(values, key=values.get)  # type: ignore[arg-type]

        row_cells = [display_names[mn]]
        for method_name in methods:
            v = values[method_name]
            bar = _score_bar(v)
            if method_name == best_method:
                row_cells.append(f"[green bold]{v:.3f}[/green bold] {bar}")
            else:
                row_cells.append(f"{v:.3f} {bar}")

        summary_tbl.add_row(*row_cells)

    console.print(summary_tbl)
    console.print()

    # ── 4. 最佳方法总结 ──
    for mn in metric_names:
        values = {m: agg[m][mn] for m in methods}
        best = max(values, key=values.get)  # type: ignore[arg-type]
        console.print(
            f"  🥇 [bold]{display_names[mn]}[/bold] 最佳: "
            f"[green bold]{best}[/green bold] ({values[best]:.3f})"
        )
    console.print()

    # ── 5. 交互模式 ──
    console.print(
        Panel(
            "[bold yellow]🎯 交互模式[/bold yellow]\n"
            "[dim]输入自定义查询，选择相关文档，查看评估结果（输入 q 退出）[/dim]",
            box=box.ROUNDED,
            expand=False,
        )
    )

    # 展示可选文档列表
    doc_tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    doc_tbl.add_column("ID", style="cyan", width=8)
    doc_tbl.add_column("标题", width=30)
    for d in docs:
        doc_tbl.add_row(d["id"], d["title"])
    console.print(doc_tbl)

    while True:
        console.print()
        query = console.input("[bold cyan]🔍 输入查询（q 退出）: [/bold cyan]").strip()
        if query.lower() == "q":
            console.print("[dim]👋 再见！[/dim]")
            break
        if not query:
            continue

        rel_input = console.input(
            "[bold cyan]📎 输入相关文档ID（逗号分隔，如 doc_01,doc_03）: [/bold cyan]"
        ).strip()
        if not rel_input:
            console.print("[yellow]⚠️  请至少输入一个相关文档ID[/yellow]")
            continue

        relevant = set(r.strip() for r in rel_input.split(",") if r.strip())
        valid_ids = {d["id"] for d in docs}
        invalid = relevant - valid_ids
        if invalid:
            console.print(f"[red]❌ 无效的文档ID: {', '.join(invalid)}[/red]")
            continue

        # 执行检索 & 评估
        bm25_ids = _search_bm25(query, bm25, docs)
        emb_ids = _search_embedding(query, model, doc_embs, docs)
        hyb_ids = _search_hybrid(query, bm25, model, doc_embs, docs)

        bm25_m = _evaluate_method(bm25_ids, relevant, K)
        emb_m = _evaluate_method(emb_ids, relevant, K)
        hyb_m = _evaluate_method(hyb_ids, relevant, K)

        result_tbl = Table(
            title="📊 自定义查询评估结果",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold yellow",
        )
        result_tbl.add_column("方法", style="cyan", width=12)
        result_tbl.add_column("Top-3 文档", width=28)
        result_tbl.add_column("MRR", justify="right", width=7)
        result_tbl.add_column(f"P@{K}", justify="right", width=7)
        result_tbl.add_column(f"R@{K}", justify="right", width=7)
        result_tbl.add_column(f"NDCG@{K}", justify="right", width=8)

        for method_name, m_dict, ranked_ids in [
            ("BM25", bm25_m, bm25_ids),
            ("Embedding", emb_m, emb_ids),
            ("Hybrid", hyb_m, hyb_ids),
        ]:
            highlighted_parts = []
            for did in ranked_ids[:K]:
                if did in relevant:
                    highlighted_parts.append(f"[green bold]{did}[/green bold]")
                else:
                    highlighted_parts.append(f"[dim]{did}[/dim]")
            top3_display = ", ".join(highlighted_parts)

            tbl_mrr = m_dict["mrr"]
            best_mrr = max(bm25_m["mrr"], emb_m["mrr"], hyb_m["mrr"])
            mrr_style = "[green bold]" if tbl_mrr == best_mrr else ""
            mrr_end = "[/green bold]" if tbl_mrr == best_mrr else ""

            result_tbl.add_row(
                method_name,
                top3_display,
                f"{m_dict['mrr']:.3f}",
                f"{m_dict['precision_at_k']:.3f}",
                f"{m_dict['recall_at_k']:.3f}",
                f"{m_dict['ndcg_at_k']:.3f}",
            )

        console.print(result_tbl)


if __name__ == "__main__":
    main()
