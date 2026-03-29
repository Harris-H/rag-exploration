"""
🔍 路线B：混合检索 Demo —— BM25 + Embedding 融合

这个脚本展示：
  1. BM25 稀疏检索（擅长精确关键词匹配）
  2. Embedding 稠密检索（擅长语义理解）
  3. RRF (Reciprocal Rank Fusion) 融合算法
  4. 三路对比：BM25 only vs Embedding only vs Hybrid
  5. 混合检索为什么在大多数场景下表现最优

混合检索是工业界公认的最佳实践 —— BM25 和 Embedding 各有所长，融合互补。

用法：uv run route_b_embedding/hybrid_search.py
"""

import json
import os
import sys
import time
import logging
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


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

    参数:
        k: 平滑参数，防止高排名文档获得过高分数（默认 60）
        bm25_weight: BM25 结果的权重
        embed_weight: Embedding 结果的权重
    """
    scores: dict[int, float] = {}

    for rank, (doc_idx, _) in enumerate(bm25_results, 1):
        scores[doc_idx] = scores.get(doc_idx, 0) + bm25_weight / (k + rank)

    for rank, (doc_idx, _) in enumerate(embed_results, 1):
        scores[doc_idx] = scores.get(doc_idx, 0) + embed_weight / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def show_triple_comparison(
    query: str,
    docs: list[dict],
    bm25_results: list[tuple[int, float]],
    embed_results: list[tuple[int, float]],
    hybrid_results: list[tuple[int, float]],
    top_k: int = 3,
):
    """三路对比展示"""
    console.print(Panel(f"[bold]查询：[/]{query}", border_style="cyan"))

    table = Table(
        title="🔀 三路检索结果对比", box=box.ROUNDED, show_lines=True, width=100
    )
    table.add_column("排名", width=4, justify="center", style="bold")
    table.add_column("🔤 BM25", width=28)
    table.add_column("🧠 Embedding", width=28)
    table.add_column("🔀 Hybrid (RRF)", width=28, style="bold green")

    for i in range(top_k):
        bm25_cell = ""
        embed_cell = ""
        hybrid_cell = ""

        if i < len(bm25_results):
            idx, score = bm25_results[i]
            bm25_cell = f"{docs[idx]['title']}\n[dim]({score:.4f})[/]"

        if i < len(embed_results):
            idx, score = embed_results[i]
            embed_cell = f"{docs[idx]['title']}\n[dim]({score:.4f})[/]"

        if i < len(hybrid_results):
            idx, score = hybrid_results[i]
            hybrid_cell = f"{docs[idx]['title']}\n[dim](RRF: {score:.6f})[/]"

        table.add_row(str(i + 1), bm25_cell, embed_cell, hybrid_cell)

    console.print(table)

    # 分析融合效果
    bm25_top = {idx for idx, _ in bm25_results[:top_k]}
    embed_top = {idx for idx, _ in embed_results[:top_k]}
    hybrid_top = {idx for idx, _ in hybrid_results[:top_k]}

    both = bm25_top & embed_top
    bm25_only = bm25_top - embed_top
    embed_only = embed_top - bm25_top
    hybrid_new = hybrid_top - bm25_top - embed_top

    analysis = []
    if both:
        titles = [docs[i]["title"] for i in both]
        analysis.append(f"[green]两者都选中：{', '.join(titles)}[/]")
    if bm25_only:
        titles = [docs[i]["title"] for i in bm25_only]
        analysis.append(f"[yellow]仅 BM25：{', '.join(titles)}[/]")
    if embed_only:
        titles = [docs[i]["title"] for i in embed_only]
        analysis.append(f"[cyan]仅 Embedding：{', '.join(titles)}[/]")
    if hybrid_new:
        titles = [docs[i]["title"] for i in hybrid_new]
        analysis.append(f"[magenta]RRF 融合后新进入：{', '.join(titles)}[/]")

    if analysis:
        console.print(
            Panel("\n".join(analysis), title="📊 融合分析", border_style="magenta")
        )


def show_weight_analysis(
    query: str,
    docs: list[dict],
    bm25_retriever: BM25Retriever,
    embed_retriever: EmbeddingRetriever,
    top_k: int = 3,
):
    """展示不同权重配置对融合结果的影响"""
    console.print(
        Panel(
            f"[bold]权重敏感性分析[/]\n"
            f"查询：{query}\n"
            f"[dim]调整 BM25 和 Embedding 的权重比例，观察结果变化[/]",
            title="⚖️ 权重分析",
            border_style="yellow",
        )
    )

    bm25_res = bm25_retriever.search(query, top_k=10)
    embed_res = embed_retriever.search(query, top_k=10)

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("权重配比\n(BM25:Embed)", width=15, justify="center", style="bold")
    for i in range(1, top_k + 1):
        table.add_column(f"Top-{i}", width=24)

    weight_configs = [
        (1.0, 0.0, "纯 BM25"),
        (0.7, 0.3, "偏 BM25"),
        (0.5, 0.5, "均衡"),
        (0.3, 0.7, "偏 Embed"),
        (0.0, 1.0, "纯 Embed"),
    ]

    for bw, ew, label in weight_configs:
        hybrid_res = rrf_fusion(bm25_res, embed_res, bm25_weight=bw, embed_weight=ew)
        row = [f"{label}\n({bw:.1f} : {ew:.1f})"]
        for i in range(top_k):
            if i < len(hybrid_res):
                idx, score = hybrid_res[i]
                row.append(docs[idx]["title"])
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


def main():
    console.print(Panel(
        "[bold]混合检索 Demo —— BM25 + Embedding 融合[/]\n"
        "使用 RRF (Reciprocal Rank Fusion) 合并词级和语义检索结果\n"
        "[dim]混合检索是工业界公认的最佳实践：精确匹配 + 语义理解 = 最优效果[/]",
        title="🔀 Route B: Hybrid Search Demo",
        border_style="bold blue",
    ))

    # 加载文档
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "..", "sample_docs", "knowledge_base.json")
    docs = load_documents(docs_path)
    console.print(f"\n[green]✅ 已加载 {len(docs)} 篇文档[/]")

    # 构建检索器
    console.print("[yellow]🔧 构建 BM25 检索器...[/]")
    bm25_retriever = BM25Retriever(docs)
    console.print("[green]✅ BM25 就绪[/]")

    try:
        console.print(f"[yellow]📥 加载 Embedding 模型: {DEFAULT_MODEL}[/]")
        embed_retriever = EmbeddingRetriever(docs)
        console.print("[green]✅ Embedding 就绪[/]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Embedding 模型加载失败：{e}[/]")
        console.print(
            "[yellow]提示：set HF_ENDPOINT=https://hf-mirror.com[/]"
        )
        sys.exit(1)

    # RRF 算法说明
    console.print(Panel(
        "[bold]RRF (Reciprocal Rank Fusion) 融合算法[/]\n\n"
        "公式：RRF_score(d) = Σ  weight_i / (k + rank_i(d))\n\n"
        "  • k = 60（平滑参数，防止第1名分数过高）\n"
        "  • rank_i(d) = 文档 d 在第 i 个检索系统中的排名\n"
        "  • weight_i = 该检索系统的权重\n\n"
        "[cyan]直觉理解[/]：\n"
        "  一篇文档如果在 BM25 和 Embedding 中排名都靠前，\n"
        "  它的 RRF 分数就会很高 —— 两个\"评委\"都认可它。\n"
        "  即使一篇文档只在一个系统中排名很高，\n"
        "  RRF 也会给予适当的分数 —— 不会因为另一个系统忽略它而被淘汰。",
        title="📐 算法原理", border_style="green",
    ))

    # 对比实验
    console.print(Panel(
        "[bold]对比实验：BM25 vs Embedding vs Hybrid[/]",
        title="🧪 对比实验", border_style="yellow",
    ))

    test_queries = [
        ("什么是RAG？", "基准查询：所有方法都能找到"),
        ("LLM 是什么", "语义测试：BM25 难匹配，Embedding 能理解 LLM ≈ 大语言模型"),
        ("文本检索排序算法", "精确匹配测试：BM25 擅长的精确关键词场景"),
        ("把长文章切成小段怎么做", "口语化改述测试：需要语义理解"),
        ("开源工具在自己电脑上运行AI模型", "综合测试：涉及 Ollama + 本地部署"),
    ]

    for query, note in test_queries:
        console.print(f"\n[dim]{note}[/]")
        bm25_res = bm25_retriever.search(query, top_k=10)
        embed_res = embed_retriever.search(query, top_k=10)
        hybrid_res = rrf_fusion(bm25_res, embed_res, bm25_weight=0.5, embed_weight=0.5)
        show_triple_comparison(query, docs, bm25_res, embed_res, hybrid_res, top_k=3)
        console.print("─" * 100)

    # 权重敏感性分析
    console.print()
    show_weight_analysis(
        "开源模型本地部署", docs, bm25_retriever, embed_retriever, top_k=3
    )

    # 总结
    console.print(Panel(
        "[bold]混合检索的核心价值[/]\n\n"
        "1. [cyan]BM25 擅长[/]：精确关键词匹配，如 \"BM25\"、\"TF-IDF\" 等专业术语\n"
        "2. [cyan]Embedding 擅长[/]：语义理解，如 \"大模型\" ≈ \"LLM\"，\"部署\" ≈ \"运行\"\n"
        "3. [cyan]Hybrid 融合[/]：两者互补，在大多数场景下都优于单一方法\n\n"
        "[yellow]工业界最佳实践：[/]\n"
        "  • Elasticsearch + 向量插件（knn_search）\n"
        "  • 先 BM25 召回 Top-100，再 Embedding 精排\n"
        "  • RRF 或 Weighted Score 融合两路结果\n\n"
        "[green]→ visualize_vectors.py 将可视化向量空间，直观理解文档分布[/]",
        title="💡 总结", border_style="cyan",
    ))

    # 交互模式
    console.print(
        "\n[bold]📝 交互模式（输入查询对比三种检索方式，输入 q 退出）[/]\n"
    )
    while True:
        console.print("[bold cyan]请输入查询：[/]")
        user_input = input("> ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            console.print("[bold]👋 再见！[/]")
            break
        if not user_input:
            continue
        bm25_res = bm25_retriever.search(user_input, top_k=10)
        embed_res = embed_retriever.search(user_input, top_k=10)
        hybrid_res = rrf_fusion(bm25_res, embed_res)
        show_triple_comparison(
            user_input, docs, bm25_res, embed_res, hybrid_res, top_k=3
        )
        console.print()


if __name__ == "__main__":
    main()
