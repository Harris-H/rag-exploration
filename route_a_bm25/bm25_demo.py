"""
🔍 BM25 极简 RAG 演示 —— 可视化理解 RAG 检索全流程

这个脚本用约 150 行代码展示了 RAG 中最核心的部分：检索（Retrieval）。
运行后你将直观看到：
  1. 文档如何被分词
  2. BM25 如何计算每篇文档的相关性分数
  3. 检索到的 Top-K 文档是什么
  4. 各文档分数的排名可视化

用法：python bm25_demo.py
"""

import json
import os
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from rank_bm25 import BM25Okapi
import jieba
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

# ============================================================
# 第 1 步：加载知识库文档
# ============================================================
def load_documents(filepath: str) -> list[dict]:
    """从 JSON 文件加载文档"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 第 2 步：文档分词（中文需要分词，英文按空格即可）
# ============================================================
def tokenize(text: str) -> list[str]:
    """使用 jieba 进行中文分词，并过滤掉单字符和标点"""
    words = jieba.lcut(text)
    # 过滤标点和单字符停用词
    stopwords = set("的了是在和与或也而且但又及其它它们这那个一不为被所有人我你他她")
    return [w for w in words if len(w) > 1 and w not in stopwords]


# ============================================================
# 第 3 步：构建 BM25 索引
# ============================================================
def build_index(docs: list[dict]) -> tuple[BM25Okapi, list[list[str]]]:
    """对所有文档分词并构建 BM25 索引"""
    tokenized_corpus = []
    for doc in docs:
        full_text = doc["title"] + " " + doc["content"]
        tokens = tokenize(full_text)
        tokenized_corpus.append(tokens)
    
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


# ============================================================
# 第 4 步：检索（RAG 的 "R"）
# ============================================================
def retrieve(bm25: BM25Okapi, query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    """BM25 检索：对 query 分词后，计算与每篇文档的相关性分数"""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # 将文档与分数打包并排序
    scored_docs = []
    for i, (doc, score) in enumerate(zip(docs, scores)):
        scored_docs.append({
            "rank": 0,
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "score": float(score),
        })
    
    # 按分数降序排列
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    for i, doc in enumerate(scored_docs):
        doc["rank"] = i + 1
    
    return scored_docs, tokenized_query


# ============================================================
# 第 5 步：可视化展示
# ============================================================
def visualize_results(query: str, tokenized_query: list[str], 
                      scored_docs: list[dict], top_k: int = 3):
    """用 Rich 库美化输出检索结果"""
    console.print()
    
    # --- RAG 流程图 ---
    flow = Panel(
        "[bold cyan]用户提问[/] → [bold yellow]分词[/] → [bold green]BM25 打分[/] → [bold magenta]排序取 Top-K[/] → [bold red]返回相关文档[/]",
        title="📊 RAG 检索流程",
        border_style="blue",
    )
    console.print(flow)
    console.print()

    # --- 显示分词结果 ---
    console.print(Panel(
        f"原始问题：[bold]{query}[/]\n分词结果：[cyan]{' | '.join(tokenized_query)}[/]",
        title="🔤 第1步：Query 分词",
        border_style="yellow",
    ))
    console.print()

    # --- BM25 全部文档评分表 ---
    score_table = Table(
        title="📈 第2步：BM25 对每篇文档的相关性评分",
        box=box.ROUNDED,
        show_lines=True,
    )
    score_table.add_column("排名", justify="center", style="bold", width=6)
    score_table.add_column("文档ID", justify="center", width=8)
    score_table.add_column("标题", width=24)
    score_table.add_column("BM25 分数", justify="right", width=12)
    score_table.add_column("分数条", width=30)

    max_score = max(d["score"] for d in scored_docs) if scored_docs else 1
    
    for doc in scored_docs:
        # 生成分数可视化条
        bar_len = int((doc["score"] / max_score) * 25) if max_score > 0 else 0
        bar = "█" * bar_len + "░" * (25 - bar_len)
        
        # Top-K 用绿色高亮
        if doc["rank"] <= top_k:
            style = "bold green"
            rank_str = f"⭐ {doc['rank']}"
        else:
            style = "dim"
            rank_str = str(doc["rank"])
        
        score_table.add_row(
            rank_str,
            doc["id"],
            Text(doc["title"], style=style),
            f"{doc['score']:.4f}",
            Text(bar, style="green" if doc["rank"] <= top_k else "dim"),
        )
    
    console.print(score_table)
    console.print()

    # --- Top-K 检索结果详情 ---
    console.print(f"[bold magenta]📄 第3步：返回 Top-{top_k} 相关文档（这就是 RAG 中传给 LLM 的上下文）[/]")
    console.print()
    
    for doc in scored_docs[:top_k]:
        content_preview = doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"]
        console.print(Panel(
            f"[bold]标题：[/]{doc['title']}\n"
            f"[bold]分数：[/][green]{doc['score']:.4f}[/]\n"
            f"[bold]内容：[/]{content_preview}",
            title=f"🏆 第 {doc['rank']} 名 - {doc['id']}",
            border_style="green",
        ))

    # --- RAG 完整流程说明 ---
    console.print()
    console.print(Panel(
        "[bold]在完整的 RAG 系统中，接下来会发生：[/]\n\n"
        f"  1. 将上面检索到的 {top_k} 篇文档拼接为上下文 (Context)\n"
        "  2. 构造 Prompt：\"根据以下参考资料回答问题：\\n{Context}\\n\\n问题：{Query}\"\n"
        "  3. 将 Prompt 发送给 LLM（如 GPT/Qwen/Llama）\n"
        "  4. LLM 基于检索到的文档生成最终回答\n\n"
        "[dim]→ 路线A 只展示了检索部分（最核心的 R），路线C 会加上完整的生成部分（G）[/]",
        title="💡 RAG = Retrieval + Generation",
        border_style="cyan",
    ))


# ============================================================
# 主程序
# ============================================================
def main():
    console.print(Panel(
        "[bold]BM25 极简 RAG 演示[/]\n"
        "这个 Demo 帮助你可视化理解 RAG 的检索过程",
        title="🔍 Route A: BM25 RAG Demo",
        border_style="bold blue",
    ))

    # 加载文档
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "..", "sample_docs", "knowledge_base.json")
    
    console.print("\n[yellow]📂 加载知识库...[/]")
    docs = load_documents(docs_path)
    console.print(f"[green]✅ 已加载 {len(docs)} 篇文档[/]\n")

    # 构建索引
    console.print("[yellow]🔧 构建 BM25 索引（分词中）...[/]")
    bm25, tokenized_corpus = build_index(docs)
    avg_tokens = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
    console.print(f"[green]✅ 索引构建完成！平均每篇文档 {avg_tokens:.0f} 个词[/]\n")

    # 预设一些示例查询
    example_queries = [
        "什么是RAG技术？它解决了什么问题？",
        "BM25算法是怎么计算的？",
        "如何在本地电脑上运行大模型？",
        "LightRAG和普通RAG有什么区别？",
        "文档分块有哪些策略？",
    ]

    console.print("[bold]📝 示例查询：[/]")
    for i, q in enumerate(example_queries, 1):
        console.print(f"  {i}. {q}")
    console.print()

    # 交互式循环
    while True:
        console.print("[bold cyan]请输入你的查询（输入 q 退出，输入数字选择示例）：[/]")
        user_input = input("> ").strip()
        
        if user_input.lower() in ("q", "quit", "exit"):
            console.print("[bold]👋 再见！[/]")
            break
        
        # 支持数字快捷选择
        if user_input.isdigit() and 1 <= int(user_input) <= len(example_queries):
            query = example_queries[int(user_input) - 1]
        elif user_input:
            query = user_input
        else:
            continue

        # 执行检索
        scored_docs, tokenized_query = retrieve(bm25, query, docs, top_k=3)
        
        # 可视化结果
        visualize_results(query, tokenized_query, scored_docs, top_k=3)
        console.print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
