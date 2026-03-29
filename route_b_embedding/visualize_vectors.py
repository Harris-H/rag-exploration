"""
🔍 路线B：向量空间可视化 —— 理解 Embedding 如何组织知识

这个脚本展示：
  1. 用 PCA 将高维 Embedding 降维到 2D，可视化文档分布
  2. 文档间的语义相似度热力图
  3. 查询向量在文档空间中的位置
  4. 直观理解为什么语义相近的文档在向量空间中聚在一起

输出：
  生成 PNG 图片文件保存在 route_b_embedding/plots/ 目录下

用法：uv run route_b_embedding/visualize_vectors.py
"""

import json
import os
import sys
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合终端环境
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    font_candidates = [
        "Microsoft YaHei",  # Windows
        "SimHei",           # Windows fallback
        "PingFang SC",      # macOS
        "Noto Sans CJK SC", # Linux
        "WenQuanYi Micro Hei",  # Linux fallback
    ]
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available = {f.name for f in fm.ttflist}
    for font in font_candidates:
        if font in available:
            rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return font
    # 无中文字体，使用英文标注
    return None


def load_documents(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_embeddings(docs: list[dict], model_name: str = DEFAULT_MODEL):
    """加载模型并计算文档 Embedding"""
    from sentence_transformers import SentenceTransformer

    console.print(f"[yellow]📥 加载 Embedding 模型: {model_name}[/]")
    model = SentenceTransformer(model_name)

    texts = [d["title"] + " " + d["content"] for d in docs]
    console.print(f"[yellow]🔧 编码 {len(texts)} 篇文档...[/]")
    doc_embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    console.print(f"[green]✅ 编码完成！维度: {doc_embeddings.shape[1]}[/]")

    return model, doc_embeddings


def plot_pca_2d(
    docs: list[dict],
    embeddings: np.ndarray,
    output_path: str,
    chinese_font: str | None,
    query_texts: list[str] | None = None,
    query_embeddings: np.ndarray | None = None,
):
    """用 PCA 降维到 2D 并绘制散点图"""
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制文档点
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=range(len(docs)), cmap="tab10", s=200, zorder=5, edgecolors="black", linewidths=0.5
    )

    # 标注文档标题
    for i, doc in enumerate(docs):
        label = doc["title"] if chinese_font else f"Doc {doc['id']}"
        ax.annotate(
            label,
            (coords_2d[i, 0], coords_2d[i, 1]),
            textcoords="offset points", xytext=(10, 8),
            fontsize=8, alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
        )

    # 绘制查询点
    if query_texts is not None and query_embeddings is not None:
        q_coords = pca.transform(query_embeddings)
        ax.scatter(
            q_coords[:, 0], q_coords[:, 1],
            marker="*", c="red", s=400, zorder=10, edgecolors="darkred", linewidths=1
        )
        for i, qt in enumerate(query_texts):
            label = qt if chinese_font else f"Q{i+1}"
            ax.annotate(
                f"Q: {label}",
                (q_coords[i, 0], q_coords[i, 1]),
                textcoords="offset points", xytext=(12, -12),
                fontsize=9, fontweight="bold", color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon", alpha=0.8),
            )

    var_ratio = pca.explained_variance_ratio_
    title = "Document Embedding Space (PCA 2D Projection)"
    if chinese_font:
        title = "文档 Embedding 空间（PCA 二维投影）"
    ax.set_title(title, fontsize=14, fontweight="bold")

    xlabel = f"PC1 ({var_ratio[0]*100:.1f}% variance)"
    ylabel = f"PC2 ({var_ratio[1]*100:.1f}% variance)"
    if chinese_font:
        xlabel = f"主成分1（解释 {var_ratio[0]*100:.1f}% 方差）"
        ylabel = f"主成分2（解释 {var_ratio[1]*100:.1f}% 方差）"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return var_ratio


def plot_similarity_heatmap(
    docs: list[dict],
    embeddings: np.ndarray,
    output_path: str,
    chinese_font: str | None,
):
    """绘制文档间语义相似度热力图"""
    sim_matrix = embeddings @ embeddings.T

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    labels = [d["title"][:10] if chinese_font else d["id"] for d in docs]
    ax.set_xticks(range(len(docs)))
    ax.set_yticks(range(len(docs)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # 在每个格子中标注相似度数值
    for i in range(len(docs)):
        for j in range(len(docs)):
            val = sim_matrix[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    title = "Semantic Similarity Heatmap"
    if chinese_font:
        title = "文档间语义相似度热力图"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return sim_matrix


def plot_query_neighbors(
    query: str,
    docs: list[dict],
    doc_embeddings: np.ndarray,
    model,
    output_path: str,
    chinese_font: str | None,
    top_k: int = 5,
):
    """绘制查询与各文档的相似度柱状图"""
    q_vec = model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ doc_embeddings.T).flatten()
    ranked_idx = np.argsort(sims)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [docs[i]["title"][:12] if chinese_font else docs[i]["id"] for i in ranked_idx]
    scores = [sims[i] for i in ranked_idx]
    colors = ["#2ecc71" if i < top_k else "#95a5a6" for i in range(len(scores))]

    bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=8)

    query_label = query if chinese_font else "Query"
    title = f'Query: "{query_label}"'
    if chinese_font:
        title = f'查询 "{query}" 与各文档的语义相似度'
    ax.set_title(title, fontsize=12, fontweight="bold")

    xlabel = "Cosine Similarity"
    if chinese_font:
        xlabel = "余弦相似度"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    console.print(Panel(
        "[bold]向量空间可视化 Demo[/]\n"
        "将高维 Embedding 向量降维到 2D，可视化文档的语义分布\n"
        "[dim]语义相近的文档在向量空间中聚在一起 —— 这就是 Embedding 的魔力[/]",
        title="📊 Route B: Vector Visualization Demo",
        border_style="bold blue",
    ))

    # 加载文档
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "..", "sample_docs", "knowledge_base.json")
    docs = load_documents(docs_path)
    console.print(f"\n[green]✅ 已加载 {len(docs)} 篇文档[/]")

    # 配置中文字体
    chinese_font = setup_chinese_font()
    if chinese_font:
        console.print(f"[green]✅ 使用中文字体: {chinese_font}[/]")
    else:
        console.print("[yellow]⚠️ 未找到中文字体，图表将使用英文标注[/]")

    # 计算 Embedding
    try:
        model, doc_embeddings = compute_embeddings(docs)
    except Exception as e:
        console.print(f"[bold red]❌ 模型加载失败：{e}[/]")
        console.print("[yellow]提示：set HF_ENDPOINT=https://hf-mirror.com[/]")
        sys.exit(1)

    # 创建输出目录
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── 图1：PCA 2D 散点图 ──
    console.print("\n[yellow]📊 生成图1: PCA 二维散点图...[/]")
    example_queries = ["什么是RAG", "LLM大语言模型", "本地部署AI"]
    query_embeddings = model.encode(example_queries, normalize_embeddings=True)
    pca_path = os.path.join(plots_dir, "pca_2d_scatter.png")
    var_ratio = plot_pca_2d(
        docs, doc_embeddings, pca_path, chinese_font,
        query_texts=example_queries, query_embeddings=query_embeddings,
    )
    console.print(
        f"[green]✅ 已保存: {pca_path}[/]\n"
        f"   PCA 前两个主成分解释了 {sum(var_ratio)*100:.1f}% 的方差"
    )

    # ── 图2：相似度热力图 ──
    console.print("\n[yellow]📊 生成图2: 语义相似度热力图...[/]")
    heatmap_path = os.path.join(plots_dir, "similarity_heatmap.png")
    sim_matrix = plot_similarity_heatmap(
        docs, doc_embeddings, heatmap_path, chinese_font
    )
    # 找出最相似的文档对
    n = len(docs)
    max_sim = 0
    max_pair = (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > max_sim:
                max_sim = sim_matrix[i, j]
                max_pair = (i, j)
    console.print(
        f"[green]✅ 已保存: {heatmap_path}[/]\n"
        f"   最相似文档对: \"{docs[max_pair[0]]['title']}\" ↔ "
        f"\"{docs[max_pair[1]]['title']}\" (相似度: {max_sim:.4f})"
    )

    # ── 图3：查询相似度柱状图 ──
    test_queries = [
        "什么是RAG技术",
        "LLM 大模型",
        "在笔记本上运行AI",
    ]
    for i, query in enumerate(test_queries, 1):
        console.print(f"\n[yellow]📊 生成图3.{i}: 查询相似度柱状图 ({query})...[/]")
        query_path = os.path.join(plots_dir, f"query_similarity_{i}.png")
        plot_query_neighbors(
            query, docs, doc_embeddings, model, query_path, chinese_font
        )
        console.print(f"[green]✅ 已保存: {query_path}[/]")

    # 总结
    console.print(Panel(
        "[bold]可视化总结[/]\n\n"
        f"生成了 {2 + len(test_queries)} 张图表，保存在 [cyan]{plots_dir}[/] 目录：\n\n"
        "  📊 pca_2d_scatter.png    — 文档在 2D 空间的分布（含查询投影）\n"
        "  📊 similarity_heatmap.png — 文档间语义相似度矩阵\n"
        + "\n".join(
            f"  📊 query_similarity_{i}.png — 查询 \"{q}\" 的相似度排名"
            for i, q in enumerate(test_queries, 1)
        ) +
        "\n\n"
        "[cyan]关键观察[/]：\n"
        "  • 语义相近的文档在 PCA 图中聚在一起\n"
        "  • 热力图中高相似度区域表示主题关联的文档对\n"
        "  • 查询向量（红色星号）落在最相关文档附近\n\n"
        "[green]→ 这就是 Embedding 模型的核心价值：将语义关系映射为空间距离[/]",
        title="💡 总结",
        border_style="cyan",
    ))


if __name__ == "__main__":
    main()
