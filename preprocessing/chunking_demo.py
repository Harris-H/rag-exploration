"""
✂️ 分块策略对比 Demo —— Chunking Strategies Comparison

这个脚本展示：
  1. 四种文档分块策略的实现与可视化
  2. 不同分块大小对检索质量的影响
  3. 重叠窗口（Overlap）对信息保留的作用

用法：uv run preprocessing/chunking_demo.py
"""

import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

console = Console()

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


# ============================================================
# 文档加载
# ============================================================
def load_documents(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 策略一：固定长度分块 (Fixed-size chunking)
# ============================================================
def chunk_fixed(text: str, chunk_size: int = 150) -> list[str]:
    """
    按固定字符数切分，不考虑语义边界。
    简单粗暴，可能在词语中间截断。
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ============================================================
# 策略二：句子边界分块 (Sentence-boundary chunking)
# ============================================================
def chunk_sentence(text: str, max_chunk_size: int = 150) -> list[str]:
    """
    在句子边界处切分（。！？；\\n），然后将句子累积到目标大小。
    永远不会在句子中间截断。
    """
    # 按句子边界拆分，保留分隔符
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
# 策略三：滑动窗口分块 (Sliding window with overlap)
# ============================================================
def chunk_sliding(text: str, chunk_size: int = 150, overlap: int = 50) -> list[str]:
    """
    固定窗口大小 + 重叠区域。
    重叠部分保留了边界处的上下文信息，避免信息丢失。
    """
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
# 策略四：递归分块 (Recursive chunking)
# ============================================================
def chunk_recursive(text: str, max_chunk_size: int = 150) -> list[str]:
    """
    多级递归分块：
      1. 先按段落分（\\n\\n）
      2. 段落太大则按句子分（。！？）
      3. 句子太大则按逗号分（，、）
      4. 仍然太大则固定切分
    """
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []

    # 依次尝试不同粒度的分隔符
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

    # 所有分隔符都无法拆分，回退到固定切分
    return chunk_fixed(text, max_chunk_size)


# ============================================================
# 展示：分块策略对比表
# ============================================================
def show_strategy_comparison(strategies: dict[str, list[str]]):
    """展示所有策略的统计对比"""
    table = Table(
        title="📊 分块策略对比",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("策略", width=20, style="bold")
    table.add_column("块数", width=8, justify="center")
    table.add_column("平均长度", width=10, justify="center")
    table.add_column("最小长度", width=10, justify="center")
    table.add_column("最大长度", width=10, justify="center")
    table.add_column("重叠", width=8, justify="center")

    strategy_meta = {
        "fixed": ("✂️ 固定长度", "无"),
        "sentence": ("📝 句子边界", "无"),
        "sliding": ("🔄 滑动窗口", "有"),
        "recursive": ("🌳 递归分块", "无"),
    }

    for name, chunks in strategies.items():
        display_name, overlap = strategy_meta.get(name, (name, "?"))
        lengths = [len(c) for c in chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        table.add_row(
            display_name,
            str(len(chunks)),
            f"{avg_len:.1f}",
            str(min_len),
            str(max_len),
            overlap,
        )

    console.print(table)


# ============================================================
# 展示：某个策略的实际分块结果
# ============================================================
def show_chunks_detail(strategy_name: str, chunks: list[str], max_show: int = 6):
    """用 Rich Panel 展示分块的实际内容"""
    colors = ["green", "blue", "magenta", "yellow", "cyan", "red"]
    display_names = {
        "fixed": "✂️ 固定长度分块",
        "sentence": "📝 句子边界分块",
        "sliding": "🔄 滑动窗口分块",
        "recursive": "🌳 递归分块",
    }
    title = display_names.get(strategy_name, strategy_name)

    console.print(
        f"\n[bold]{title}[/] — 前 {min(max_show, len(chunks))} 个分块"
        f"（共 {len(chunks)} 块）："
    )

    for i, chunk in enumerate(chunks[:max_show]):
        color = colors[i % len(colors)]
        preview = chunk[:120] + ("..." if len(chunk) > 120 else "")
        console.print(Panel(
            preview,
            title=f"块 {i + 1} ({len(chunk)} 字符)",
            border_style=color,
            width=80,
        ))


# ============================================================
# 展示：检索质量对比
# ============================================================
def show_retrieval_comparison(query: str, strategy_results: dict):
    """展示不同分块策略下的检索结果对比"""
    console.print(Panel(
        f"[bold]查询：[/]{query}",
        border_style="cyan",
    ))

    tables = []
    for name, info in strategy_results.items():
        display_names = {
            "fixed": "✂️ 固定长度",
            "sentence": "📝 句子边界",
            "sliding": "🔄 滑动窗口",
            "recursive": "🌳 递归分块",
        }
        t = Table(
            title=display_names.get(name, name),
            box=box.ROUNDED,
            width=44,
            show_lines=True,
        )
        t.add_column("#", width=3, justify="center")
        t.add_column("内容摘要", width=22)
        t.add_column("相似度", width=8, justify="right")

        for i, (chunk, score) in enumerate(info["top_matches"], 1):
            preview = chunk[:20] + "..."
            # 分数条形图
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            t.add_row(str(i), preview, f"{score:.4f}")

        tables.append(t)

    # 两两并排展示
    if len(tables) >= 2:
        console.print(Columns(tables[:2], padding=(0, 2)))
    if len(tables) >= 4:
        console.print(Columns(tables[2:4], padding=(0, 2)))

    # 分数条形图
    console.print("\n[bold]📈 Top-1 相似度对比：[/]")
    for name, info in strategy_results.items():
        display_names = {
            "fixed": "✂️ 固定长度",
            "sentence": "📝 句子边界",
            "sliding": "🔄 滑动窗口",
            "recursive": "🌳 递归分块",
        }
        if info["top_matches"]:
            score = info["top_matches"][0][1]
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            display = display_names.get(name, name)
            console.print(f"  {display:<12} {bar} {score:.4f}")


# ============================================================
# 检索测试：对每种策略的分块做向量检索
# ============================================================
def run_retrieval_test(query: str, strategies: dict[str, list[str]], model) -> dict:
    """对每种策略的分块执行向量检索，返回 top-3 匹配"""
    q_vec = model.encode([query], normalize_embeddings=True)
    results = {}

    for name, chunks in strategies.items():
        if not chunks:
            results[name] = {"top_matches": [], "num_chunks": 0}
            continue

        chunk_vecs = model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False
        )
        sims = (q_vec @ chunk_vecs.T).flatten()
        top_indices = np.argsort(sims)[::-1][:3]

        results[name] = {
            "top_matches": [(chunks[i], float(sims[i])) for i in top_indices],
            "num_chunks": len(chunks),
        }

    return results


# ============================================================
# Main
# ============================================================
def main():
    console.print(Panel(
        "[bold]分块策略对比 Demo[/]\n"
        "展示四种常见的文档分块策略及其对检索质量的影响\n"
        "[dim]分块是 RAG 流水线中至关重要的预处理步骤[/]",
        title="✂️ Chunking Strategies Comparison",
        border_style="bold blue",
    ))

    # 加载文档
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "..", "sample_docs", "knowledge_base.json")
    docs = load_documents(docs_path)
    console.print(f"\n[green]✅ 已加载 {len(docs)} 篇文档[/]")

    # 拼接测试文本：doc_01（RAG 基础）+ doc_05（分块）+ doc_09（混合检索）
    target_ids = ["doc_01", "doc_05", "doc_09"]
    selected = [d for d in docs if d["id"] in target_ids]
    if len(selected) < len(target_ids):
        # 回退：取前 3 篇
        selected = docs[:3]
    test_text = "\n\n".join(d["title"] + "。" + d["content"] for d in selected)
    console.print(
        f"[yellow]📄 测试文本：拼接了 {len(selected)} 篇文档，"
        f"共 {len(test_text)} 字符[/]\n"
    )

    # ── 阶段一：应用四种分块策略 ──
    console.print("[bold cyan]═══ 阶段一：分块策略对比 ═══[/]\n")

    chunk_size = 150
    overlap = 50

    t0 = time.perf_counter()
    strategies = {
        "fixed": chunk_fixed(test_text, chunk_size),
        "sentence": chunk_sentence(test_text, chunk_size),
        "sliding": chunk_sliding(test_text, chunk_size, overlap),
        "recursive": chunk_recursive(test_text, chunk_size),
    }
    chunk_time = time.perf_counter() - t0

    show_strategy_comparison(strategies)
    console.print(f"[dim]分块耗时：{chunk_time * 1000:.1f}ms[/]\n")

    # ── 阶段二：展示各策略的分块细节 ──
    console.print("[bold cyan]═══ 阶段二：分块内容预览 ═══[/]\n")

    for name in ["fixed", "sentence", "sliding", "recursive"]:
        show_chunks_detail(name, strategies[name], max_show=4)
        console.print()

    # 展示滑动窗口的重叠效果
    if len(strategies["sliding"]) >= 2:
        c1, c2 = strategies["sliding"][0], strategies["sliding"][1]
        overlap_text = ""
        # 找到相邻块的重叠部分
        for length in range(min(len(c1), len(c2)), 0, -1):
            if c1.endswith(c2[:length]):
                overlap_text = c2[:length]
                break
        if overlap_text:
            console.print(Panel(
                f"[bold]块 1 尾部 ↔ 块 2 头部 重叠内容：[/]\n"
                f"[green]{overlap_text[:100]}{'...' if len(overlap_text) > 100 else ''}[/]\n"
                f"[dim]重叠长度：{len(overlap_text)} 字符 — 这保留了边界处的上下文信息[/]",
                title="🔗 滑动窗口重叠示例",
                border_style="yellow",
            ))
            console.print()

    # ── 阶段三：检索质量对比 ──
    console.print("[bold cyan]═══ 阶段三：检索质量对比 ═══[/]\n")

    # 加载 Embedding 模型
    console.print(f"[yellow]📥 加载 Embedding 模型: {DEFAULT_MODEL}[/]")
    try:
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        model = SentenceTransformer(DEFAULT_MODEL)
        load_time = time.perf_counter() - t0
        console.print(f"[green]✅ 模型加载完成！耗时 {load_time:.1f}s[/]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Embedding 模型加载失败：{e}[/]")
        console.print(
            "[yellow]提示：中国大陆用户可设置环境变量加速下载：\n"
            "  set HF_ENDPOINT=https://hf-mirror.com\n"
            "  然后重新运行此脚本[/]"
        )
        sys.exit(1)

    # 预设测试查询
    test_queries = [
        "RAG 如何解决大模型的幻觉问题",
        "文档分块对检索效果有什么影响",
        "向量检索和关键词检索如何结合",
    ]

    for query in test_queries:
        t0 = time.perf_counter()
        results = run_retrieval_test(query, strategies, model)
        elapsed = time.perf_counter() - t0
        show_retrieval_comparison(query, results)
        console.print(f"[dim]检索耗时：{elapsed * 1000:.1f}ms[/]\n")
        console.print("─" * 80)

    # 原理总结
    console.print(Panel(
        "[bold]分块策略选择指南[/]\n\n"
        "  ✂️ [cyan]固定长度[/]：实现简单，但可能截断语义单元\n"
        "     适用：快速原型、对质量要求不高的场景\n\n"
        "  📝 [cyan]句子边界[/]：保持句子完整性，块大小不均匀\n"
        "     适用：需要保持语义完整性的通用场景\n\n"
        "  🔄 [cyan]滑动窗口[/]：重叠区域保留边界上下文，块数最多\n"
        "     适用：信息密集文档、不能遗漏边界信息的场景\n\n"
        "  🌳 [cyan]递归分块[/]：自适应粒度，尊重文档结构\n"
        "     适用：结构化文档、LangChain 默认策略\n\n"
        "[green]→ 实际项目中推荐 [bold]递归分块[/green] 作为默认策略，"
        "结合 [bold]滑动窗口[/] 处理长段落[/]",
        title="💡 策略选择总结",
        border_style="cyan",
    ))

    # ── 交互模式 ──
    console.print(
        "\n[bold]📝 交互模式（输入问题，对比四种分块策略的检索效果，输入 q 退出）[/]\n"
    )
    while True:
        console.print("[bold cyan]请输入查询：[/]")
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]👋 再见！[/]")
            break
        if user_input.lower() in ("q", "quit", "exit"):
            console.print("[bold]👋 再见！[/]")
            break
        if not user_input:
            continue

        t0 = time.perf_counter()
        results = run_retrieval_test(user_input, strategies, model)
        elapsed = time.perf_counter() - t0
        show_retrieval_comparison(user_input, results)
        console.print(f"[dim]检索耗时：{elapsed * 1000:.1f}ms[/]\n")


if __name__ == "__main__":
    main()
