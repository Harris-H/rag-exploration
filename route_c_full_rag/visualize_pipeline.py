"""
📊 路线C-3：RAG 流程可视化 —— 对比有无 RAG 的效果差异

这个脚本生成可视化图表：
  1. RAG 全流程示意图（检索 → 上下文拼接 → LLM 生成）
  2. 有 RAG vs 无 RAG 的回答质量对比
  3. 检索结果相关度分布图
  4. 不同检索模式的响应时间对比

输出文件保存到 route_c_full_rag/plots/ 目录

前置要求：
  - Ollama 已安装并运行
  - 已下载模型：qwen2.5:3b, nomic-embed-text

用法：uv run route_c_full_rag/visualize_pipeline.py
"""

import json
import os
import sys
import time

# 确保本地 Ollama 请求不走系统代理（避免 502 Bad Gateway）
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1"

import numpy as np
import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from rich.console import Console
from rich.panel import Panel

console = Console()

# ─── 配置 ────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# 绕过代理访问本地 Ollama
http_client = httpx.Client(proxy=None, timeout=120)

# ─── 中文字体配置 ────────────────────────────────────
def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    import platform
    if platform.system() == "Windows":
        candidates = ["Microsoft YaHei", "SimHei", "SimSun"]
    else:
        candidates = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]

    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available = {f.name for f in fm.ttflist}

    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            console.print(f"[green]✓[/green] 使用字体: {font}")
            return font

    console.print("[yellow]⚠ 未找到中文字体，图表中文可能显示为方块[/yellow]")
    return None


# ─── Ollama API ──────────────────────────────────────
def ollama_embed(texts: list[str]) -> list[list[float]]:
    resp = http_client.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def ollama_generate(prompt: str, system: str = "") -> tuple[str, float]:
    """生成回答，返回 (回答, 耗时秒数)"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    resp = http_client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
    )
    resp.raise_for_status()
    elapsed = time.time() - t0
    answer = resp.json()["message"]["content"]
    return answer, elapsed


# ─── 知识库 & 检索 ───────────────────────────────────
def load_kb():
    kb_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "knowledge_base.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve(query: str, docs: list[dict], doc_embeddings: list, top_k: int = 3):
    query_vec = ollama_embed([query])[0]
    scores = [cosine_similarity(query_vec, dv) for dv in doc_embeddings]
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return [(docs[i], s) for i, s in ranked[:top_k]], scores


# ─── 图1: RAG 流程架构图 ─────────────────────────────
def plot_rag_pipeline(save_path: str):
    """绘制 RAG 全流程示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.set_axis_off()
    fig.suptitle("RAG 全流程架构图", fontsize=16, fontweight="bold", y=0.98)

    # 定义组件
    components = [
        (1.0, 3.0, 2.0, 1.2, "用户问题", "#4ECDC4", "white"),
        (4.0, 4.2, 2.0, 1.2, "向量检索", "#45B7D1", "white"),
        (4.0, 1.6, 2.0, 1.2, "知识库", "#96CEB4", "white"),
        (7.5, 3.0, 2.5, 1.2, "Prompt\n拼接", "#FFEAA7", "black"),
        (11.0, 3.0, 2.0, 1.2, "LLM\n生成回答", "#DDA0DD", "white"),
    ]

    for x, y, w, h, label, color, tcolor in components:
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="gray", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=tcolor)

    # 箭头
    arrows = [
        (3.0, 3.6, 4.0, 4.6),    # 问题 → 向量检索
        (4.5, 2.8, 4.5, 4.2),    # 知识库 → 向量检索
        (6.0, 4.8, 7.5, 3.9),    # 向量检索 → Prompt拼接
        (3.0, 3.3, 7.5, 3.3),    # 问题 → Prompt拼接
        (10.0, 3.6, 11.0, 3.6),  # Prompt拼接 → LLM
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2))

    # 标注
    ax.text(3.5, 5.0, "① 编码查询", fontsize=9, color="#666", style="italic")
    ax.text(4.0, 1.0, "② 检索相关文档", fontsize=9, color="#666", style="italic")
    ax.text(8.2, 4.7, "③ 上下文 + 问题", fontsize=9, color="#666", style="italic")
    ax.text(10.5, 4.5, "④ 生成回答", fontsize=9, color="#666", style="italic")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── 图2: 检索相关度分布 ─────────────────────────────
def plot_retrieval_scores(queries, all_scores, doc_titles, save_path):
    """每个查询的文档相关度得分分布"""
    n_queries = len(queries)
    fig, axes = plt.subplots(1, n_queries, figsize=(6 * n_queries, 5))
    if n_queries == 1:
        axes = [axes]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(doc_titles)))

    for ax, query, scores in zip(axes, queries, all_scores):
        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = [scores[i] for i in sorted_idx]
        sorted_titles = [doc_titles[i] for i in sorted_idx]

        # 缩短标题
        short_titles = [t[:8] + "..." if len(t) > 8 else t for t in sorted_titles]

        bar_colors = ["#4ECDC4" if s > 0.5 else "#FFB347" if s > 0.3 else "#FF6B6B"
                      for s in sorted_scores]

        bars = ax.barh(range(len(sorted_scores)), sorted_scores, color=bar_colors, edgecolor="white")
        ax.set_yticks(range(len(sorted_scores)))
        ax.set_yticklabels(short_titles, fontsize=8)
        ax.set_xlabel("余弦相似度")
        ax.set_title(f"查询: {query[:15]}...", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

        # 添加数值标签
        for bar, score in zip(bars, sorted_scores):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f"{score:.3f}", va="center", fontsize=8)

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor="#4ECDC4", label="高相关 (>0.5)"),
        mpatches.Patch(facecolor="#FFB347", label="中相关 (0.3~0.5)"),
        mpatches.Patch(facecolor="#FF6B6B", label="低相关 (<0.3)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("文档检索相关度分布", fontsize=14, fontweight="bold", y=1.08)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── 图3: RAG vs 无 RAG 对比 ─────────────────────────
def plot_rag_vs_direct(comparisons, save_path):
    """对比 RAG 和直接生成的响应时间与回答长度"""
    queries = [c["query"][:12] + "..." for c in comparisons]
    rag_times = [c["rag_time"] for c in comparisons]
    direct_times = [c["direct_time"] for c in comparisons]
    rag_lens = [len(c["rag_answer"]) for c in comparisons]
    direct_lens = [len(c["direct_answer"]) for c in comparisons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 响应时间对比
    x = np.arange(len(queries))
    width = 0.35
    bars1 = ax1.bar(x - width/2, rag_times, width, label="有 RAG", color="#4ECDC4", edgecolor="white")
    bars2 = ax1.bar(x + width/2, direct_times, width, label="无 RAG", color="#FF6B6B", edgecolor="white")
    ax1.set_xlabel("查询")
    ax1.set_ylabel("响应时间 (秒)")
    ax1.set_title("响应时间对比", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 回答长度对比
    bars3 = ax2.bar(x - width/2, rag_lens, width, label="有 RAG", color="#4ECDC4", edgecolor="white")
    bars4 = ax2.bar(x + width/2, direct_lens, width, label="无 RAG", color="#FF6B6B", edgecolor="white")
    ax2.set_xlabel("查询")
    ax2.set_ylabel("回答长度 (字符)")
    ax2.set_title("回答详细度对比", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(queries, fontsize=8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("RAG vs 直接生成对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── 图4: 检索命中热力图 ─────────────────────────────
def plot_query_doc_heatmap(queries, all_scores, doc_titles, save_path):
    """查询-文档 相似度热力图"""
    matrix = np.array(all_scores)
    short_titles = [t[:10] + ".." if len(t) > 10 else t for t in doc_titles]
    short_queries = [q[:12] + ".." if len(q) > 12 else q for q in queries]

    fig, ax = plt.subplots(figsize=(max(8, len(doc_titles) * 0.8), max(4, len(queries) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(short_titles)))
    ax.set_xticklabels(short_titles, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(short_queries)))
    ax.set_yticklabels(short_queries, fontsize=9)

    # 数值标签
    for i in range(len(queries)):
        for j in range(len(doc_titles)):
            text_color = "white" if matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                   fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label="余弦相似度")
    ax.set_title("查询-文档 相似度热力图", fontweight="bold", fontsize=13)
    ax.set_xlabel("文档")
    ax.set_ylabel("查询")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── 主流程 ──────────────────────────────────────────
def main():
    console.print(Panel.fit(
        "[bold cyan]📊 RAG 流程可视化[/bold cyan]\n"
        "[dim]生成 RAG 流程图、检索分布、RAG vs 无RAG 对比等可视化图表[/dim]",
        border_style="bright_blue",
    ))

    # 检查 Ollama
    try:
        resp = http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        console.print("[green]✓[/green] Ollama 连接正常")
    except Exception as e:
        console.print(f"[red]✗[/red] 无法连接 Ollama: {e}")
        sys.exit(1)

    setup_chinese_font()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 图1: RAG 流程架构图（不需要 LLM） ──
    console.print("\n[bold]📈 1/4 绘制 RAG 流程架构图...[/bold]")
    plot_rag_pipeline(os.path.join(PLOTS_DIR, "01_rag_pipeline.png"))
    console.print("[green]✓[/green] 已保存 plots/01_rag_pipeline.png")

    # ── 加载知识库 & 编码 ──
    docs = load_kb()
    doc_titles = [d["title"] for d in docs]
    console.print(f"\n[bold]⚙️ 编码 {len(docs)} 篇知识库文档...[/bold]")
    t0 = time.time()
    texts = [f"{d['title']}：{d['content']}" for d in docs]
    doc_embeddings = ollama_embed(texts)
    console.print(f"[green]✓[/green] 文档编码完成 ({time.time()-t0:.1f}s)")

    # ── 查询集 ──
    queries = [
        "LightRAG 框架的核心特点是什么？",
        "如何在本地电脑运行大语言模型？",
        "混合检索策略有什么优势？",
        "什么是文档分块？有哪些策略？",
    ]

    # ── 检索所有查询 ──
    console.print(f"\n[bold]🔍 检索 {len(queries)} 个查询...[/bold]")
    all_scores = []
    all_top_results = []
    for q in queries:
        top_results, scores = retrieve(q, docs, doc_embeddings)
        all_scores.append(scores)
        all_top_results.append(top_results)
        console.print(f"  {q[:20]}... → Top1: {top_results[0][0]['title']} ({top_results[0][1]:.3f})")

    # ── 图2: 检索相关度分布 ──
    console.print("\n[bold]📈 2/4 绘制检索相关度分布...[/bold]")
    plot_retrieval_scores(queries, all_scores, doc_titles,
                         os.path.join(PLOTS_DIR, "02_retrieval_scores.png"))
    console.print("[green]✓[/green] 已保存 plots/02_retrieval_scores.png")

    # ── 图4: 热力图 ──
    console.print("\n[bold]📈 3/4 绘制查询-文档热力图...[/bold]")
    plot_query_doc_heatmap(queries, all_scores, doc_titles,
                          os.path.join(PLOTS_DIR, "03_similarity_heatmap.png"))
    console.print("[green]✓[/green] 已保存 plots/03_similarity_heatmap.png")

    # ── RAG vs 无 RAG 对比（需要 LLM 生成） ──
    console.print("\n[bold]📈 4/4 RAG vs 无 RAG 对比（需要 LLM 生成，较慢）...[/bold]")
    comparisons = []
    for q, top_results in zip(queries, all_top_results):
        console.print(f"  处理: {q[:25]}...")

        # 构建 RAG prompt
        context = "\n\n".join(
            f"[参考文档{i+1}] {doc['title']}\n{doc['content']}"
            for i, (doc, _) in enumerate(top_results)
        )
        rag_prompt = f"根据以下参考文档回答问题，用中文简洁回答。\n\n{context}\n\n问题：{q}"
        rag_answer, rag_time = ollama_generate(rag_prompt, system="你是知识库助手，基于文档回答。")

        # 无 RAG
        direct_answer, direct_time = ollama_generate(
            f"请用中文简洁回答：{q}", system="你是AI助手。"
        )

        comparisons.append({
            "query": q,
            "rag_answer": rag_answer,
            "rag_time": rag_time,
            "direct_answer": direct_answer,
            "direct_time": direct_time,
        })
        console.print(f"    RAG: {rag_time:.1f}s ({len(rag_answer)}字) | 直接: {direct_time:.1f}s ({len(direct_answer)}字)")

    plot_rag_vs_direct(comparisons, os.path.join(PLOTS_DIR, "04_rag_vs_direct.png"))
    console.print("[green]✓[/green] 已保存 plots/04_rag_vs_direct.png")

    # ── 汇总 ──
    console.print()
    console.print(Panel(
        f"所有图表已保存到 [bold]{PLOTS_DIR}[/bold] 目录：\n"
        "  📊 01_rag_pipeline.png      — RAG 全流程架构图\n"
        "  📊 02_retrieval_scores.png   — 检索相关度分布\n"
        "  📊 03_similarity_heatmap.png — 查询-文档相似度热力图\n"
        "  📊 04_rag_vs_direct.png      — RAG vs 无RAG 对比",
        title="✅ 可视化完成",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
