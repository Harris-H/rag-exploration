"""
🤖 路线C-1：极简 RAG Demo —— 从零手搓 RAG 流程

这个脚本展示 RAG 的本质：
  1. 检索阶段：使用 Route B 的 Embedding 模型找到相关文档
  2. 增强阶段：将检索到的文档拼接成上下文 Prompt
  3. 生成阶段：调用 Ollama 本地 LLM 生成回答

不依赖任何 RAG 框架，帮助理解 RAG 的核心机制：
  「检索到的文档 + 用户问题 → 拼成 Prompt → 喂给 LLM」

前置要求：
  - Ollama 已安装并运行（ollama serve）
  - 已下载模型：ollama pull qwen2.5:3b

用法：uv run route_c_full_rag/simple_rag_demo.py
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
import jieba
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

console = Console()

# ─── 配置 ────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
TOP_K = 3

# 绕过代理访问本地 Ollama（避免系统代理拦截 localhost 请求）
http_client = httpx.Client(proxy=None, timeout=120)
http_stream_client = httpx.Client(proxy=None, timeout=120)

# ─── 知识库加载 ──────────────────────────────────────
def load_knowledge_base() -> list[dict]:
    kb_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "knowledge_base.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Ollama API 封装 ─────────────────────────────────
def ollama_embed(texts: list[str]) -> list[list[float]]:
    """调用 Ollama embedding API 获取文本向量"""
    resp = http_client.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def ollama_generate(prompt: str, system: str = "") -> str:
    """调用 Ollama 生成 API，流式输出"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    collected = []
    with http_client.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": True},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                collected.append(chunk)
                console.print(chunk, end="", highlight=False)
    console.print()
    return "".join(collected)


# ─── 检索器 ──────────────────────────────────────────
class SimpleRetriever:
    """极简向量检索器：Ollama Embedding + 余弦相似度"""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.texts = [f"{d['title']}：{d['content']}" for d in docs]
        console.print("[dim]正在编码知识库文档...[/dim]")
        t0 = time.time()
        self.embeddings = np.array(ollama_embed(self.texts))
        elapsed = time.time() - t0
        console.print(f"[green]✓[/green] {len(docs)} 篇文档编码完成 ({elapsed:.1f}s), 维度={self.embeddings.shape[1]}")

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[dict, float]]:
        query_vec = np.array(ollama_embed([query])[0])
        # 余弦相似度
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        scores = np.dot(self.embeddings, query_vec) / (norms + 1e-10)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in top_indices]


class BM25Retriever:
    """BM25 稀疏检索器"""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.texts = [f"{d['title']} {d['content']}" for d in docs]
        tokenized = [list(jieba.cut(t)) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[dict, float]]:
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in top_indices]


# ─── RAG Prompt 构建 ─────────────────────────────────
def build_rag_prompt(query: str, context_docs: list[tuple[dict, float]]) -> str:
    """将检索到的文档 + 用户问题拼成 RAG Prompt"""
    context_parts = []
    for i, (doc, score) in enumerate(context_docs, 1):
        context_parts.append(f"[参考文档{i}] {doc['title']}\n{doc['content']}")

    context_text = "\n\n".join(context_parts)

    return f"""请根据以下参考文档回答用户的问题。如果参考文档中没有相关信息，请如实说明。

---参考文档---
{context_text}
---参考文档结束---

用户问题：{query}

请用中文回答，条理清晰，适当引用参考文档中的内容。"""


SYSTEM_PROMPT = "你是一个专业的 RAG 知识库助手，基于提供的参考文档来回答用户问题。回答要准确、简洁，不要编造参考文档中没有的信息。"


# ─── 主流程 ──────────────────────────────────────────
def check_ollama() -> bool:
    """检查 Ollama 服务是否可用"""
    try:
        resp = http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        console.print(f"[green]✓[/green] Ollama 服务已连接，可用模型: {', '.join(models)}")

        need_models = {LLM_MODEL, EMBED_MODEL}
        # 检查模型是否存在（允许部分匹配，如 "qwen2.5:3b" 匹配 "qwen2.5:3b"）
        available = set()
        for m in models:
            for need in need_models:
                if need in m or m.startswith(need.split(":")[0]):
                    available.add(need)
        missing = need_models - available
        if missing:
            console.print(f"[red]✗[/red] 缺少模型: {', '.join(missing)}")
            console.print(f"  请运行: ollama pull {' && ollama pull '.join(missing)}")
            return False
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] 无法连接 Ollama ({OLLAMA_BASE_URL}): {e}")
        console.print("  请确保 Ollama 正在运行: ollama serve")
        return False


def display_retrieval_results(results: list[tuple[dict, float]], label: str):
    """展示检索结果"""
    table = Table(title=f"📚 {label} 检索结果 (Top {len(results)})", box=box.ROUNDED)
    table.add_column("排名", style="bold cyan", width=4)
    table.add_column("文档标题", style="bold white", width=22)
    table.add_column("相似度", style="yellow", width=8)
    table.add_column("内容片段", style="dim", max_width=50)
    for i, (doc, score) in enumerate(results, 1):
        snippet = doc["content"][:80] + "..." if len(doc["content"]) > 80 else doc["content"]
        table.add_row(str(i), doc["title"], f"{score:.4f}", snippet)
    console.print(table)


def run_rag_query(query: str, retriever: SimpleRetriever, bm25: BM25Retriever):
    """执行一次完整的 RAG 查询"""
    console.rule(f"[bold blue]🔍 查询: {query}")

    # Step 1: 向量检索
    console.print("\n[bold]Step 1: 向量语义检索[/bold]")
    t0 = time.time()
    embed_results = retriever.search(query, TOP_K)
    embed_time = time.time() - t0
    display_retrieval_results(embed_results, "Embedding")
    console.print(f"[dim]检索耗时: {embed_time*1000:.0f}ms[/dim]")

    # Step 2: BM25 检索（对比）
    console.print("\n[bold]Step 2: BM25 关键词检索（对比）[/bold]")
    bm25_results = bm25.search(query, TOP_K)
    display_retrieval_results(bm25_results, "BM25")

    # Step 3: 构建 RAG Prompt
    console.print("\n[bold]Step 3: 构建 RAG Prompt[/bold]")
    rag_prompt = build_rag_prompt(query, embed_results)
    prompt_preview = rag_prompt[:200] + "..." if len(rag_prompt) > 200 else rag_prompt
    console.print(Panel(prompt_preview, title="Prompt 预览", border_style="dim"))

    # Step 4: LLM 生成（有 RAG 上下文）
    console.print(f"\n[bold]Step 4: LLM 生成回答（有 RAG 上下文, 模型={LLM_MODEL}）[/bold]")
    t0 = time.time()
    rag_answer = ollama_generate(rag_prompt, system=SYSTEM_PROMPT)
    rag_time = time.time() - t0
    console.print(f"[dim]生成耗时: {rag_time:.1f}s[/dim]")

    # Step 5: 无 RAG 直接生成（对比）
    console.print(f"\n[bold]Step 5: 无 RAG 直接问 LLM（对比）[/bold]")
    t0 = time.time()
    direct_answer = ollama_generate(
        f"请用中文简洁回答：{query}",
        system="你是一个AI助手，请简洁回答问题。",
    )
    direct_time = time.time() - t0
    console.print(f"[dim]生成耗时: {direct_time:.1f}s[/dim]")

    # 总结对比
    console.print()
    compare = Table(title="📊 RAG vs 无 RAG 对比", box=box.DOUBLE_EDGE, show_lines=True)
    compare.add_column("方式", style="bold", width=14)
    compare.add_column("回答摘要", max_width=60)
    compare.add_column("耗时", style="yellow", width=8)
    rag_summary = rag_answer[:150] + "..." if len(rag_answer) > 150 else rag_answer
    direct_summary = direct_answer[:150] + "..." if len(direct_answer) > 150 else direct_answer
    compare.add_row("✅ 有 RAG", rag_summary, f"{rag_time:.1f}s")
    compare.add_row("❌ 无 RAG", direct_summary, f"{direct_time:.1f}s")
    console.print(compare)


def main():
    console.print(Panel.fit(
        "[bold cyan]🤖 极简 RAG Demo —— 从零手搓 RAG 流程[/bold cyan]\n"
        "[dim]不依赖任何框架，展示 RAG 的本质：检索 → 拼 Prompt → 生成[/dim]",
        border_style="bright_blue",
    ))

    # 检查 Ollama
    if not check_ollama():
        sys.exit(1)

    # 加载知识库
    docs = load_knowledge_base()
    console.print(f"\n[green]✓[/green] 加载知识库: {len(docs)} 篇文档")

    # 初始化检索器
    retriever = SimpleRetriever(docs)
    bm25 = BM25Retriever(docs)

    # 预设演示查询
    demo_queries = [
        "LightRAG 和传统 RAG 有什么区别？",
        "如何在个人电脑上运行大语言模型？",
        "什么是混合检索？它为什么比单一检索方法更好？",
    ]

    console.print(Panel(
        "\n".join(f"  {i}. {q}" for i, q in enumerate(demo_queries, 1)),
        title="💡 预设演示查询",
        border_style="yellow",
    ))

    # 运行演示查询
    for q in demo_queries:
        run_rag_query(q, retriever, bm25)
        console.print()

    # 交互模式
    console.rule("[bold green]💬 进入交互模式（输入 q 退出）")
    while True:
        try:
            query = console.input("\n[bold cyan]你的问题 > [/bold cyan]").strip()
            if not query or query.lower() in ("q", "quit", "exit"):
                console.print("[yellow]再见！👋[/yellow]")
                break
            run_rag_query(query, retriever, bm25)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见！👋[/yellow]")
            break


if __name__ == "__main__":
    main()
