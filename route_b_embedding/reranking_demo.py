"""
🔄 路线B：Reranking Demo —— Cross-Encoder 重排序

这个脚本展示：
  1. 初次检索（BM25 + Embedding）获取候选文档
  2. 使用 Cross-Encoder 对候选文档重新打分排序
  3. 对比重排序前后的结果变化

Cross-Encoder 与 Bi-Encoder 的区别：
  - Bi-Encoder：分别编码 query 和 doc，再算相似度（速度快，适合召回）
  - Cross-Encoder：将 query 和 doc 拼接后一起编码，直接输出相关性分数（更准确，适合精排）

用法：uv run route_b_embedding/reranking_demo.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

console = Console()

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")


def load_documents(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_zh(text: str) -> list[str]:
    stopwords = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in stopwords]


# ============================================================
# BM25 检索器
# ============================================================
class BM25Retriever:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        corpus = [tokenize_zh(d["title"] + " " + d["content"]) for d in docs]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10):
        """返回 [(doc_index, score), ...] 按分数降序"""
        scores = self.bm25.get_scores(tokenize_zh(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ============================================================
# Embedding 检索器
# ============================================================
class EmbeddingRetriever:
    def __init__(self, docs: list[dict], model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self.model = SentenceTransformer(model_name)
        texts = [d["title"] + " " + d["content"] for d in docs]
        self.doc_embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 10):
        """返回 [(doc_index, score), ...] 按分数降序"""
        q_vec = self.model.encode([query], normalize_embeddings=True)
        sims = (q_vec @ self.doc_embeddings.T).flatten()
        ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ============================================================
# RRF 混合检索
# ============================================================
def rrf_fusion(
    bm25_results: list[tuple[int, float]],
    embed_results: list[tuple[int, float]],
    k: int = 60,
    bm25_weight: float = 0.5,
    embed_weight: float = 0.5,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) 融合算法。
    RRF 公式：score(d) = Σ weight_i / (k + rank_i(d))
    """
    scores: dict[int, float] = {}
    for rank, (doc_idx, _) in enumerate(bm25_results, 1):
        scores[doc_idx] = scores.get(doc_idx, 0) + bm25_weight / (k + rank)
    for rank, (doc_idx, _) in enumerate(embed_results, 1):
        scores[doc_idx] = scores.get(doc_idx, 0) + embed_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# Cross-Encoder 重排序器
# ============================================================
class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, docs: list[dict], doc_indices: list[int]
    ) -> list[tuple[int, float]]:
        """
        对候选文档进行 Cross-Encoder 打分并重排序。
        返回 [(doc_index, ce_score), ...] 按 CE 分数降序。
        """
        pairs = [
            (query, docs[idx]["title"] + " " + docs[idx]["content"])
            for idx in doc_indices
        ]
        scores = self.model.predict(pairs)
        # 将分数与原始索引配对，按分数降序排列
        scored = list(zip(doc_indices, scores.tolist()))
        return sorted(scored, key=lambda x: x[1], reverse=True)


# ============================================================
# 可视化：重排序前后对比
# ============================================================
def show_rerank_comparison(
    query: str,
    docs: list[dict],
    before_ranked: list[tuple[int, float]],
    after_ranked: list[tuple[int, float]],
    top_k: int = 5,
):
    """并排展示重排序前后的结果"""
    console.print(Panel(f"[bold]查询：[/]{query}", border_style="cyan"))

    # 构建排名变化映射：doc_idx -> (before_rank, after_rank)
    before_rank_map = {idx: rank for rank, (idx, _) in enumerate(before_ranked, 1)}
    after_rank_map = {idx: rank for rank, (idx, _) in enumerate(after_ranked, 1)}

    # === 表1：初次检索结果 ===
    t1 = Table(
        title="📊 初次检索结果 (RRF 混合)", box=box.ROUNDED, width=48, show_lines=True
    )
    t1.add_column("排名", width=4, justify="center", style="bold")
    t1.add_column("文档标题", width=22)
    t1.add_column("RRF 分数", width=10, justify="right")
    t1.add_column("分数条", width=8)

    max_rrf = before_ranked[0][1] if before_ranked else 1.0
    for i, (idx, score) in enumerate(before_ranked[:top_k]):
        bar_len = int((score / max_rrf) * 8) if max_rrf > 0 else 0
        bar = "█" * bar_len + "░" * (8 - bar_len)
        t1.add_row(str(i + 1), docs[idx]["title"], f"{score:.6f}", f"[yellow]{bar}[/]")

    # === 表2：Cross-Encoder 重排序后 ===
    t2 = Table(
        title="🔄 Cross-Encoder 重排序后", box=box.ROUNDED, width=48, show_lines=True
    )
    t2.add_column("排名", width=4, justify="center", style="bold")
    t2.add_column("文档标题", width=22)
    t2.add_column("CE 分数", width=10, justify="right")
    t2.add_column("排名变化", width=8, justify="center")

    for i, (idx, score) in enumerate(after_ranked[:top_k]):
        new_rank = i + 1
        old_rank = before_rank_map.get(idx, top_k + 1)
        change = old_rank - new_rank
        if change > 0:
            change_str = f"[bold green]↑{change}[/]"
        elif change < 0:
            change_str = f"[bold red]↓{abs(change)}[/]"
        else:
            change_str = "[dim]═[/]"
        t2.add_row(str(new_rank), docs[idx]["title"], f"{score:.4f}", change_str)

    console.print(Columns([t1, t2], padding=(0, 2)))

    # 分析重排序效果
    before_top = [idx for idx, _ in before_ranked[:3]]
    after_top = [idx for idx, _ in after_ranked[:3]]

    promoted = []
    demoted = []
    for idx, _ in after_ranked[:top_k]:
        new_rank = after_rank_map[idx]
        old_rank = before_rank_map.get(idx, top_k + 1)
        if new_rank < old_rank:
            promoted.append((docs[idx]["title"], old_rank, new_rank))
        elif new_rank > old_rank:
            demoted.append((docs[idx]["title"], old_rank, new_rank))

    analysis = []
    if promoted:
        for title, old_r, new_r in promoted:
            analysis.append(f"[green]↑ {title}：第{old_r}名 → 第{new_r}名[/]")
    if demoted:
        for title, old_r, new_r in demoted:
            analysis.append(f"[red]↓ {title}：第{old_r}名 → 第{new_r}名[/]")

    if before_top != after_top:
        analysis.append(
            "\n[cyan]💡 Cross-Encoder 通过深度语义理解调整了文档排名[/]"
        )
    else:
        analysis.append(
            "\n[dim]初次检索排名已较准确，Cross-Encoder 确认了排序[/]"
        )

    if analysis:
        console.print(
            Panel(
                "\n".join(analysis),
                title="📈 重排序效果分析",
                border_style="magenta",
            )
        )


def show_timing(
    bm25_ms: float, embed_ms: float, rrf_ms: float, rerank_ms: float
):
    """展示各阶段耗时"""
    total = bm25_ms + embed_ms + rrf_ms + rerank_ms
    table = Table(title="⏱️ 各阶段耗时", box=box.ROUNDED, width=50)
    table.add_column("阶段", width=20, style="bold")
    table.add_column("耗时", width=12, justify="right")
    table.add_column("占比", width=10, justify="right")

    stages = [
        ("BM25 检索", bm25_ms),
        ("Embedding 检索", embed_ms),
        ("RRF 融合", rrf_ms),
        ("Cross-Encoder 重排序", rerank_ms),
    ]
    for name, ms in stages:
        pct = (ms / total * 100) if total > 0 else 0
        table.add_row(name, f"{ms:.1f} ms", f"{pct:.1f}%")
    table.add_row("[bold]总计[/]", f"[bold]{total:.1f} ms[/]", "[bold]100%[/]")
    console.print(table)


def run_rerank_query(
    query: str,
    docs: list[dict],
    bm25_retriever: BM25Retriever,
    embed_retriever: EmbeddingRetriever,
    reranker: CrossEncoderReranker,
    top_k: int = 5,
    show_timing_info: bool = True,
):
    """执行完整的检索 + 重排序流程"""
    # Stage 1a: BM25 检索
    t0 = time.perf_counter()
    bm25_res = bm25_retriever.search(query, top_k=10)
    bm25_ms = (time.perf_counter() - t0) * 1000

    # Stage 1b: Embedding 检索
    t0 = time.perf_counter()
    embed_res = embed_retriever.search(query, top_k=10)
    embed_ms = (time.perf_counter() - t0) * 1000

    # Stage 2: RRF 融合
    t0 = time.perf_counter()
    hybrid_ranked = rrf_fusion(bm25_res, embed_res, bm25_weight=0.5, embed_weight=0.5)
    rrf_ms = (time.perf_counter() - t0) * 1000

    # 取 hybrid top-k 候选
    candidate_indices = [idx for idx, _ in hybrid_ranked[:top_k]]

    # Stage 3: Cross-Encoder 重排序
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, candidate_indices)
    rerank_ms = (time.perf_counter() - t0) * 1000

    # 展示对比
    show_rerank_comparison(query, docs, hybrid_ranked[:top_k], reranked, top_k=top_k)
    if show_timing_info:
        show_timing(bm25_ms, embed_ms, rrf_ms, rerank_ms)


def main():
    console.print(Panel(
        "[bold]Reranking Demo —— Cross-Encoder 重排序[/]\n"
        "使用 Cross-Encoder 对初次检索结果进行精细重排序\n"
        "[dim]Cross-Encoder 将 query 和 doc 拼接后联合编码，比 Bi-Encoder 更准确[/]",
        title="🔄 Route B: Reranking Demo",
        border_style="bold blue",
    ))

    # 加载文档
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "..", "sample_docs", "knowledge_base.json")
    docs = load_documents(docs_path)
    console.print(f"\n[green]✅ 已加载 {len(docs)} 篇文档[/]")

    # 构建 BM25 检索器
    console.print("[yellow]🔧 构建 BM25 检索器...[/]")
    bm25_retriever = BM25Retriever(docs)
    console.print("[green]✅ BM25 就绪[/]")

    # 加载 Embedding 模型
    try:
        console.print(f"[yellow]📥 加载 Embedding 模型: {DEFAULT_MODEL}[/]")
        embed_retriever = EmbeddingRetriever(docs)
        console.print("[green]✅ Embedding 就绪[/]")
    except Exception as e:
        console.print(f"[bold red]❌ Embedding 模型加载失败：{e}[/]")
        console.print(
            "[yellow]提示：set HF_ENDPOINT=https://hf-mirror.com[/]"
        )
        sys.exit(1)

    # 加载 Cross-Encoder 重排序模型
    try:
        console.print(f"[yellow]📥 加载 Cross-Encoder 模型: {RERANKER_MODEL}[/]")
        console.print("[dim]（首次运行需下载模型，约 400MB，请耐心等待...）[/]")
        t0 = time.perf_counter()
        reranker = CrossEncoderReranker()
        load_time = time.perf_counter() - t0
        console.print(f"[green]✅ Cross-Encoder 就绪！加载耗时 {load_time:.1f}s[/]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Cross-Encoder 模型加载失败：{e}[/]")
        console.print(
            "[yellow]提示：set HF_ENDPOINT=https://hf-mirror.com[/]"
        )
        sys.exit(1)

    # Cross-Encoder 原理说明
    console.print(Panel(
        "[bold]Cross-Encoder 重排序原理[/]\n\n"
        "传统检索流程（两阶段）：\n"
        "  [cyan]Stage 1[/]：BM25 + Embedding 召回 Top-N 候选文档（速度快）\n"
        "  [cyan]Stage 2[/]：Cross-Encoder 对候选精排（更准确）\n\n"
        "Cross-Encoder vs Bi-Encoder：\n"
        "  • [yellow]Bi-Encoder[/]：query 和 doc 分别编码 → 余弦相似度\n"
        "    → 速度快（可预计算文档向量），但交互信息有限\n\n"
        "  • [green]Cross-Encoder[/]：[query, SEP, doc] 一起输入 Transformer\n"
        "    → 速度慢（无法预计算），但能捕捉深层交互语义\n\n"
        f"本 Demo 使用模型：{RERANKER_MODEL}\n"
        "[dim]这是一个针对中文优化的 Cross-Encoder 重排序模型[/]",
        title="📐 算法原理", border_style="green",
    ))

    # 对比实验
    console.print(Panel(
        "[bold]对比实验：RRF 混合检索 vs Cross-Encoder 重排序[/]",
        title="🧪 对比实验", border_style="yellow",
    ))

    test_queries = [
        ("什么是向量数据库", "基准查询：向量数据库相关文档应排在前面"),
        ("如何在本地运行大模型", "语义理解测试：需要理解 '大模型' ≈ 'LLM'，'本地运行' ≈ 'Ollama 部署'"),
        ("RAG解决了什么问题", "概念理解测试：需要深度理解 RAG 的动机和价值"),
        ("文本检索排序算法", "精确匹配测试：BM25、TF-IDF 相关文档"),
        ("把长文章切成小段怎么做", "口语化改述测试：需要理解 '切成小段' = '文档分块'"),
    ]

    for query, note in test_queries:
        console.print(f"\n[dim]{note}[/]")
        run_rerank_query(
            query, docs, bm25_retriever, embed_retriever, reranker,
            top_k=5, show_timing_info=False,
        )
        console.print("─" * 100)

    # 耗时分析（单独展示一次）
    console.print("\n[bold]⏱️ 性能分析（以最后一个查询为例）[/]\n")
    run_rerank_query(
        "开源工具在自己电脑上运行AI模型", docs,
        bm25_retriever, embed_retriever, reranker,
        top_k=5, show_timing_info=True,
    )

    # 总结
    console.print(Panel(
        "[bold]Cross-Encoder 重排序的核心价值[/]\n\n"
        "1. [cyan]更准确[/]：Cross-Encoder 联合编码 query 和 doc，能捕捉深层语义交互\n"
        "2. [cyan]两阶段[/]：先用 BM25+Embedding 快速召回，再用 Cross-Encoder 精排\n"
        "3. [cyan]工业实践[/]：Google、Bing 等搜索引擎都采用多阶段检索+重排序架构\n\n"
        "[yellow]重排序使得语义更相关的文档排名上升，关键词匹配但语义不相关的文档排名下降[/]\n\n"
        "[green]适用场景：[/]\n"
        "  • 候选文档较少（<100）时效果显著\n"
        "  • 对检索精度要求高的场景（问答、对话）\n"
        "  • RAG 系统中，重排序后再送入 LLM 可显著提升回答质量",
        title="💡 总结", border_style="cyan",
    ))

    # 交互模式
    console.print(
        "\n[bold]📝 交互模式（输入查询对比重排序效果，输入 q 退出）[/]"
    )
    console.print(
        "[dim]示例：什么是向量数据库 / 如何在本地运行大模型 / RAG解决了什么问题[/]\n"
    )
    while True:
        console.print("[bold cyan]请输入查询：[/]")
        user_input = input("> ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            console.print("[bold]👋 再见！[/]")
            break
        if not user_input:
            continue
        run_rerank_query(
            user_input, docs, bm25_retriever, embed_retriever, reranker,
            top_k=5, show_timing_info=True,
        )
        console.print()


if __name__ == "__main__":
    main()
