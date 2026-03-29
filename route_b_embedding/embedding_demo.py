"""
🔍 路线B：语义检索 Demo —— 真实 Embedding 模型 vs BM25

这个脚本展示：
  1. 使用 sentence-transformers 加载预训练 Embedding 模型（BGE-small-zh）
  2. 将文本编码为 512 维稠密语义向量
  3. 通过余弦相似度实现语义检索
  4. 对比 BM25（词级匹配）vs Embedding（语义匹配）的差异
  5. 展示语义检索的核心优势：同义词/近义词/语义理解

环境变量：
  HF_ENDPOINT=https://hf-mirror.com  # 中国大陆用户可设置 HuggingFace 镜像加速下载
  EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5  # 自定义模型名称

用法：uv run route_b_embedding/embedding_demo.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

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


def load_documents(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_zh(text: str) -> list[str]:
    stopwords = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in stopwords]


# ============================================================
# BM25 检索（词级匹配 —— 作为对比基线）
# ============================================================
def build_bm25(docs: list[dict]) -> BM25Okapi:
    corpus = [tokenize_zh(d["title"] + " " + d["content"]) for d in docs]
    return BM25Okapi(corpus)


def retrieve_bm25(bm25: BM25Okapi, query: str, docs: list[dict], top_k: int = 3):
    scores = bm25.get_scores(tokenize_zh(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(docs[i], float(s)) for i, s in ranked[:top_k]]


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2 归一化向量"""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms


# ============================================================
# 神经网络 Embedding 语义检索
# ============================================================
class EmbeddingSearch:
    """
    使用 sentence-transformers 预训练模型将文本编码为稠密向量。
    与 TF-IDF 的关键区别：
      - TF-IDF：基于字符统计，只能匹配字面重叠
      - Embedding：基于神经网络，能理解语义（"大模型" ≈ "LLM"）
    """

    def __init__(self, docs: list[dict], model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self.model_name = model_name
        console.print(f"[yellow]📥 加载 Embedding 模型: {model_name}[/]")
        console.print("[dim]（首次运行需下载模型，约 100MB，请耐心等待...）[/]")

        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - t0
        console.print(f"[green]✅ 模型加载完成！耗时 {load_time:.1f}s[/]")

        # 编码所有文档
        texts = [d["title"] + " " + d["content"] for d in docs]
        console.print(f"[yellow]🔧 编码 {len(texts)} 篇文档为语义向量...[/]")
        t0 = time.time()
        self.doc_embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        encode_time = time.time() - t0
        self.dim = self.doc_embeddings.shape[1]
        console.print(
            f"[green]✅ 编码完成！维度: {self.dim}, 耗时 {encode_time:.2f}s[/]"
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3):
        q_vec = self.encode_query(query)
        sims = (q_vec @ self.doc_embeddings.T).flatten()
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return [(self.docs[i], float(s)) for i, s in ranked[:top_k]]

    def get_all_scores(self, query: str) -> list[float]:
        q_vec = self.encode_query(query)
        return (q_vec @ self.doc_embeddings.T).flatten().tolist()


def show_comparison(query: str, bm25_results, embed_results):
    """并排展示 BM25 和 Embedding 检索结果"""
    t1 = Table(
        title="🔤 BM25 (词级匹配)", box=box.SIMPLE, width=46, show_lines=True
    )
    t1.add_column("#", width=3, justify="center")
    t1.add_column("文档", width=22)
    t1.add_column("BM25分数", width=10, justify="right")
    for i, (doc, score) in enumerate(bm25_results, 1):
        t1.add_row(str(i), doc["title"], f"{score:.4f}")

    t2 = Table(
        title="🧠 Embedding (语义匹配)", box=box.SIMPLE, width=46, show_lines=True
    )
    t2.add_column("#", width=3, justify="center")
    t2.add_column("文档", width=22)
    t2.add_column("余弦相似度", width=10, justify="right")
    for i, (doc, score) in enumerate(embed_results, 1):
        t2.add_row(str(i), doc["title"], f"{score:.4f}")

    console.print(Panel(f"[bold]查询：[/]{query}", border_style="cyan"))
    console.print(Columns([t1, t2], padding=(0, 2)))

    # 差异分析
    bm25_titles = [d["title"] for d, _ in bm25_results]
    embed_titles = [d["title"] for d, _ in embed_results]
    only_embed = [t for t in embed_titles if t not in bm25_titles]
    only_bm25 = [t for t in bm25_titles if t not in embed_titles]
    if only_embed or only_bm25:
        diff = ""
        if only_embed:
            diff += f"[green]🧠 Embedding 独有：{', '.join(only_embed)}[/]\n"
        if only_bm25:
            diff += f"[yellow]🔤 BM25 独有：{', '.join(only_bm25)}[/]\n"
        console.print(Panel(diff.strip(), title="🔍 差异分析", border_style="magenta"))


def show_embedding_info(embed_search: EmbeddingSearch):
    """展示 Embedding 向量空间的关键信息"""
    mat = embed_search.doc_embeddings
    sparsity = (np.abs(mat) < 1e-6).sum() / mat.size * 100
    console.print(Panel(
        f"[bold]Embedding 向量空间信息[/]\n\n"
        f"  模型：{embed_search.model_name}\n"
        f"  文档数量：{mat.shape[0]}\n"
        f"  向量维度：{mat.shape[1]}  (每篇文档 = {mat.shape[1]} 维空间中的一个点)\n"
        f"  接近零的分量占比：{sparsity:.1f}%  ([bold green]稠密向量[/])\n"
        f"  平均向量模长：{np.mean(np.linalg.norm(mat, axis=1)):.4f}  (已归一化)\n\n"
        f"[dim]与 TF-IDF 稀疏向量（~5000维，>95% 为零）相比，\n"
        f"Embedding 稠密向量（{mat.shape[1]}维，几乎无零值）能捕捉深层语义。[/]",
        title="🧠 Embedding 向量空间", border_style="green",
    ))


def show_similarity_matrix(embed_search: EmbeddingSearch):
    """展示文档间的语义相似度矩阵"""
    mat = embed_search.doc_embeddings
    sims = (mat @ mat.T).round(3)
    docs = embed_search.docs

    table = Table(title="📊 文档间语义相似度矩阵", box=box.ROUNDED)
    table.add_column("", width=14, style="bold")
    short_names = [d["title"][:8] for d in docs]
    for name in short_names:
        table.add_column(name, width=8, justify="center")

    for i, doc in enumerate(docs):
        row = []
        for j in range(len(docs)):
            val = sims[i][j]
            if i == j:
                row.append("[dim]1.000[/]")
            elif val > 0.7:
                row.append(f"[bold green]{val:.3f}[/]")
            elif val > 0.5:
                row.append(f"[yellow]{val:.3f}[/]")
            else:
                row.append(f"[dim]{val:.3f}[/]")
        table.add_row(doc["title"][:12], *row)

    console.print(table)
    console.print(
        "[dim]绿色 = 高语义相似(>0.7)，黄色 = 中等(>0.5)，灰色 = 低[/]\n"
    )


def show_semantic_advantage(bm25: BM25Okapi, embed_search: EmbeddingSearch, docs):
    """展示语义检索独有的优势：同义词/近义词理解"""
    console.print(Panel(
        "[bold]语义检索 vs 词级检索 —— 关键差异演示[/]\n"
        "以下查询使用同义词或改述，BM25 难以匹配，但 Embedding 能理解",
        title="🎯 语义优势展示", border_style="yellow",
    ))

    semantic_queries = [
        ("LLM 是什么", "测试：'LLM' 与 '大语言模型' 是同义词"),
        ("人工智能文档搜索", "测试：'搜索' ≈ '检索'，'人工智能' ≈ 'AI'"),
        ("如何让模型回答更准确", "测试：语义上关联 RAG、幻觉问题"),
        ("把长文章切成小段", "测试：这是对 '文档分块' 的口语化改述"),
        ("开源模型跑在自己电脑上", "测试：'跑在电脑上' ≈ '本地部署'"),
    ]

    for query, note in semantic_queries:
        console.print(f"\n[dim]{note}[/]")
        bm25_res = retrieve_bm25(bm25, query, docs, top_k=3)
        embed_res = embed_search.search(query, top_k=3)
        show_comparison(query, bm25_res, embed_res)
        console.print("─" * 80)


def main():
    console.print(Panel(
        "[bold]语义检索 Demo —— 真实 Embedding 模型[/]\n"
        "使用 sentence-transformers 预训练模型，对比 BM25 词级匹配 vs Embedding 语义匹配\n"
        "[dim]Embedding 能理解同义词、近义词和语义关联，这是传统检索做不到的[/]",
        title="🧠 Route B: Semantic Search Demo",
        border_style="bold blue",
    ))

    # 加载文档
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "..", "sample_docs", "knowledge_base.json")
    docs = load_documents(docs_path)
    console.print(f"\n[green]✅ 已加载 {len(docs)} 篇文档[/]")

    # 构建 BM25 基线
    console.print("[yellow]🔧 构建 BM25 索引...[/]")
    bm25 = build_bm25(docs)
    console.print("[green]✅ BM25 索引就绪[/]\n")

    # 加载 Embedding 模型
    try:
        embed_search = EmbeddingSearch(docs)
    except Exception as e:
        console.print(f"[bold red]❌ Embedding 模型加载失败：{e}[/]")
        console.print(
            "[yellow]提示：中国大陆用户可设置环境变量加速下载：\n"
            "  set HF_ENDPOINT=https://hf-mirror.com\n"
            "  然后重新运行此脚本[/]"
        )
        sys.exit(1)

    console.print()

    # 展示向量空间信息
    show_embedding_info(embed_search)
    console.print()

    # 展示文档间语义相似度
    show_similarity_matrix(embed_search)

    # 语义优势演示
    show_semantic_advantage(bm25, embed_search, docs)

    # 原理说明
    console.print(Panel(
        "[bold]Embedding 语义检索的核心原理[/]\n\n"
        "1. [cyan]预训练[/]：模型在海量文本上学习语言的语义表示\n"
        "2. [cyan]编码[/]：每篇文档 → 一个稠密向量（本 Demo: {dim} 维）\n"
        "3. [cyan]查询编码[/]：用户问题 → 同一向量空间中的点\n"
        "4. [cyan]检索[/]：计算余弦相似度，语义越近分数越高\n\n"
        "[yellow]关键优势：[/]\n"
        '  • "大模型" 和 "LLM" → 向量空间中距离很近\n'
        '  • "本地部署" 和 "在电脑上运行" → 语义等价\n'
        '  • BM25 只能匹配相同的字/词，Embedding 理解含义\n\n'
        "[green]→ hybrid_search.py 将展示如何融合 BM25 + Embedding 获得最佳效果[/]"
        .format(dim=embed_search.dim),
        title="💡 原理总结",
        border_style="cyan",
    ))

    # 交互模式
    console.print(
        "\n[bold]📝 交互模式（输入问题对比 BM25 vs Embedding，输入 q 退出）[/]\n"
    )
    while True:
        console.print("[bold cyan]请输入查询：[/]")
        user_input = input("> ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            console.print("[bold]👋 再见！[/]")
            break
        if not user_input:
            continue
        bm25_res = retrieve_bm25(bm25, user_input, docs, top_k=3)
        embed_res = embed_search.search(user_input, top_k=3)
        show_comparison(user_input, bm25_res, embed_res)
        console.print()


if __name__ == "__main__":
    main()
