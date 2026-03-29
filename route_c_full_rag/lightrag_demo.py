"""
🧠 路线C-2：LightRAG 完整演示 —— 知识图谱增强的 RAG

这个脚本展示 LightRAG 框架的完整能力：
  1. 文档索引：自动抽取实体和关系，构建知识图谱
  2. 多模式查询：naive / local / global / hybrid / mix 五种检索模式
  3. 知识图谱：展示自动构建的实体-关系图谱
  4. 增量更新：支持新增文档而无需重建索引

技术栈：
  - LLM: Ollama + qwen2.5:3b（本地运行）
  - Embedding: Ollama + nomic-embed-text
  - RAG 框架: LightRAG（知识图谱 + 向量双路检索）

前置要求：
  - Ollama 已安装并运行（ollama serve）
  - 已下载模型：ollama pull qwen2.5:3b && ollama pull nomic-embed-text

用法：uv run route_c_full_rag/lightrag_demo.py
"""

import asyncio
import json
import os
import sys
import shutil
import time

import numpy as np

# 确保本地 Ollama 请求不走系统代理（避免 502 Bad Gateway）
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1"

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
WORKING_DIR = os.path.join(os.path.dirname(__file__), "lightrag_data")
EMBED_DIM = 768  # nomic-embed-text 默认维度


# ─── Ollama 检查 ─────────────────────────────────────
def check_ollama() -> bool:
    import httpx
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5, proxy=None)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        console.print(f"[green]✓[/green] Ollama 可用模型: {', '.join(models)}")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] 无法连接 Ollama: {e}")
        console.print("  请运行: ollama serve")
        return False


# ─── 知识库 ──────────────────────────────────────────
def load_knowledge_base() -> str:
    """加载知识库并拼接成长文本（LightRAG 接受纯文本输入）"""
    kb_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "knowledge_base.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # 将每篇文档格式化为带标题的文本段落
    sections = []
    for doc in docs:
        sections.append(f"## {doc['title']}\n\n{doc['content']}")

    return "\n\n---\n\n".join(sections)


# ─── LightRAG 初始化 ─────────────────────────────────
async def init_lightrag(fresh: bool = False):
    """初始化 LightRAG 实例"""
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc
    import ollama as ollama_sdk

    if fresh and os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
        console.print("[yellow]已清除旧索引数据[/yellow]")
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 自定义 embedding 函数，绕过 LightRAG 内置 ollama_embed 的维度硬编码
    async def custom_embed(texts: list[str]) -> np.ndarray:
        client = ollama_sdk.AsyncClient(host=OLLAMA_BASE_URL)
        try:
            data = await client.embed(model=EMBED_MODEL, input=texts)
            return np.array(data["embeddings"])
        finally:
            await client._client.aclose()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_BASE_URL,
            "options": {"num_ctx": 4096},
            "timeout": 0,  # Ollama 客户端不超时
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=custom_embed,
        ),
        # CPU 推理较慢，增加超时（默认 180s 不够）
        default_llm_timeout=600,
    )

    await rag.initialize_storages()

    return rag


# ─── 索引阶段 ────────────────────────────────────────
async def index_documents(rag, text: str):
    """将文档索引到 LightRAG"""
    console.print("\n[bold]📝 Step 1: 文档索引[/bold]")
    console.print("[dim]LightRAG 将自动：分块 → 抽取实体/关系 → 构建知识图谱 → 生成向量索引[/dim]")
    console.print("[dim]（首次运行需要 LLM 处理每个文档块，可能需要数分钟...）[/dim]")

    t0 = time.time()
    await rag.ainsert(text)
    elapsed = time.time() - t0
    console.print(f"[green]✓[/green] 索引完成! 耗时 {elapsed:.1f}s")


# ─── 知识图谱展示 ─────────────────────────────────────
def show_knowledge_graph(rag):
    """展示 LightRAG 构建的知识图谱"""
    console.print("\n[bold]🕸️ Step 2: 知识图谱[/bold]")

    # 尝试读取知识图谱数据
    kg_path = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(kg_path):
        console.print("[yellow]知识图谱文件尚未生成[/yellow]")
        return

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(kg_path)
        root = tree.getroot()
        ns = {"g": "http://graphml.graphstudio.org/xmlns"}

        # 统计节点和边
        nodes = root.findall(".//{http://graphml.graphstudio.org/xmlns}node")
        edges = root.findall(".//{http://graphml.graphstudio.org/xmlns}edge")

        # 如果 namespace 不匹配，尝试无命名空间
        if not nodes:
            nodes = root.findall(".//{http://graphml.graphstudio.org/xmlns}node")
        if not nodes:
            # 尝试通用命名空间
            for elem in root.iter():
                if elem.tag.endswith("node"):
                    nodes.append(elem)
                elif elem.tag.endswith("edge"):
                    edges.append(elem)

        console.print(f"  实体(节点)数: [cyan]{len(nodes)}[/cyan]")
        console.print(f"  关系(边)数: [cyan]{len(edges)}[/cyan]")

        # 展示部分实体
        if nodes:
            table = Table(title="📋 部分实体节点（前10个）", box=box.ROUNDED)
            table.add_column("ID", style="bold cyan", max_width=20)
            table.add_column("属性", style="dim", max_width=50)

            for node in nodes[:10]:
                node_id = node.get("id", "?")
                attrs = []
                for data in node:
                    key = data.get("key", "")
                    text = (data.text or "")[:60]
                    if text:
                        attrs.append(f"{key}={text}")
                table.add_row(node_id, "; ".join(attrs[:3]))
            console.print(table)

    except Exception as e:
        console.print(f"[yellow]解析知识图谱时出错: {e}[/yellow]")
        # 至少展示文件大小
        size = os.path.getsize(kg_path)
        console.print(f"  图谱文件大小: {size/1024:.1f}KB")


# ─── 多模式查询 ──────────────────────────────────────
async def query_comparison(rag, query: str):
    """对比 LightRAG 的多种查询模式"""
    from lightrag import QueryParam

    console.rule(f"[bold blue]🔍 查询: {query}")

    modes = [
        ("naive", "朴素模式 —— 纯向量检索，不使用知识图谱"),
        ("local", "局部模式 —— 基于实体的局部图检索"),
        ("global", "全局模式 —— 基于社区摘要的全局检索"),
        ("hybrid", "混合模式 —— local + global 结合"),
        ("mix", "综合模式 —— 向量 + 知识图谱全方位检索"),
    ]

    results = {}
    for mode, desc in modes:
        console.print(f"\n[bold]{mode.upper()}[/bold] - {desc}")
        t0 = time.time()
        try:
            answer = await rag.aquery(query, param=QueryParam(mode=mode))
            elapsed = time.time() - t0
            answer = answer or "(无结果)"
            results[mode] = (answer, elapsed)

            # 截取显示
            if len(answer) > 300:
                display = answer[:300] + "..."
            else:
                display = answer
            console.print(Panel(display, border_style="dim", title=f"⏱ {elapsed:.1f}s"))
        except Exception as e:
            elapsed = time.time() - t0
            console.print(f"  [red]查询失败: {e}[/red]")
            results[mode] = (f"错误: {e}", elapsed)

    # 汇总表
    console.print()
    summary = Table(title="📊 五种模式对比", box=box.DOUBLE_EDGE, show_lines=True)
    summary.add_column("模式", style="bold", width=10)
    summary.add_column("回答摘要", max_width=55)
    summary.add_column("耗时", style="yellow", width=8)
    for mode, _ in modes:
        answer, elapsed = results.get(mode, ("N/A", 0))
        snippet = (answer[:120] + "...") if len(answer) > 120 else answer
        summary.add_row(mode, snippet, f"{elapsed:.1f}s")
    console.print(summary)


# ─── 增量更新演示 ─────────────────────────────────────
async def demo_incremental_update(rag):
    """演示增量更新能力"""
    console.print("\n[bold]📥 Step 4: 增量更新演示[/bold]")
    console.print("[dim]向已有索引中新增一篇文档，无需重建...[/dim]")

    new_doc = """## 向量量化技术

向量量化（Vector Quantization）是一种压缩高维向量的技术，在大规模向量检索中至关重要。
常见方法包括：乘积量化（PQ，Product Quantization）将高维向量分成多个子空间分别量化，
标量量化（SQ，Scalar Quantization）直接降低每个维度的精度。
量化后的向量占用存储减少10-100倍，检索速度也大幅提升，但会引入一定的精度损失。
FAISS、Milvus 等向量数据库都内置了多种量化方案。"""

    t0 = time.time()
    await rag.ainsert(new_doc)
    elapsed = time.time() - t0
    console.print(f"[green]✓[/green] 增量更新完成! 耗时 {elapsed:.1f}s")

    # 查询新增的内容
    from lightrag import QueryParam
    console.print("\n[dim]验证：查询新增文档的内容...[/dim]")
    answer = await rag.aquery("向量量化技术有哪些方法？", param=QueryParam(mode="mix"))
    if answer:
        console.print(Panel(answer[:400] if len(answer) > 400 else answer, title="增量更新后的查询结果", border_style="green"))
    else:
        console.print("[yellow]查询未返回结果（可能需要更多上下文）[/yellow]")


# ─── 主流程 ──────────────────────────────────────────
async def async_main():
    console.print(Panel.fit(
        "[bold cyan]🧠 LightRAG 完整演示[/bold cyan]\n"
        "[dim]知识图谱增强的 RAG —— 自动抽取实体关系，五种检索模式对比[/dim]",
        border_style="bright_blue",
    ))

    # 检查 Ollama
    if not check_ollama():
        sys.exit(1)

    # 判断是否需要重新索引
    index_exists = os.path.exists(os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml"))
    if index_exists:
        console.print(f"\n[yellow]检测到已有索引数据: {WORKING_DIR}[/yellow]")
        console.print("  使用已有索引（跳过索引阶段）")
        console.print("  如需重建，请删除 lightrag_data 目录或设置环境变量 REBUILD=1")
        fresh = os.environ.get("REBUILD", "0") == "1"
    else:
        fresh = True

    # 初始化 LightRAG
    console.print("\n[bold]⚙️ 初始化 LightRAG...[/bold]")
    rag = await init_lightrag(fresh=fresh)
    console.print(f"[green]✓[/green] LightRAG 已就绪 (LLM={LLM_MODEL}, Embed={EMBED_MODEL})")

    # 索引文档
    if fresh or not index_exists:
        text = load_knowledge_base()
        console.print(f"[green]✓[/green] 加载知识库: {len(text)} 字符")
        await index_documents(rag, text)

    # 展示知识图谱
    show_knowledge_graph(rag)

    # 多模式查询对比
    console.print("\n[bold]🔍 Step 3: 多模式查询对比[/bold]")
    console.print("[dim]注意：CPU 推理较慢，每种模式约需 30-60s[/dim]")
    demo_queries = [
        "LightRAG 框架的核心特点是什么？它和传统 RAG 有何不同？",
    ]

    for q in demo_queries:
        await query_comparison(rag, q)
        console.print()

    # 增量更新
    await demo_incremental_update(rag)

    # 交互模式
    console.rule("[bold green]💬 进入交互模式（输入 q 退出，输入 mode:xxx 切换模式）")
    console.print("[dim]默认使用 mix 模式，支持: naive, local, global, hybrid, mix[/dim]")

    from lightrag import QueryParam
    current_mode = "mix"

    while True:
        try:
            query = console.input(f"\n[bold cyan][{current_mode}] 你的问题 > [/bold cyan]").strip()
            if not query or query.lower() in ("q", "quit", "exit"):
                console.print("[yellow]再见！👋[/yellow]")
                break

            # 切换模式
            if query.lower().startswith("mode:"):
                new_mode = query.split(":", 1)[1].strip()
                if new_mode in ("naive", "local", "global", "hybrid", "mix"):
                    current_mode = new_mode
                    console.print(f"[green]✓[/green] 切换到 {current_mode} 模式")
                else:
                    console.print(f"[red]无效模式，可选: naive, local, global, hybrid, mix[/red]")
                continue

            t0 = time.time()
            answer = await rag.aquery(query, param=QueryParam(mode=current_mode))
            elapsed = time.time() - t0
            answer = answer or "(无结果)"
            console.print(Panel(answer, title=f"[{current_mode}] ⏱ {elapsed:.1f}s", border_style="green"))

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见！👋[/yellow]")
            break


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
