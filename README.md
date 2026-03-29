# RAG 技术探索 —— 从 BM25 到完整 LLM RAG

可视化理解 RAG（检索增强生成）检索流程的学习项目。提供 **Web 可视化前端** + **FastAPI 后端** + **命令行脚本** 三种交互方式。

## 快速开始

### 1. 后端环境

```bash
# 需要 Python 3.11+（建议使用独立 Python，不建议 Anaconda）、uv 包管理器
uv sync
```

> 💡 首次运行会自动下载 Embedding 模型（BGE-small-zh，约 50MB）。  
> 中国大陆用户如下载缓慢，可设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`

### 2. 前端环境

```bash
cd web
npm install
```

> 需要 Node.js 18+

### 3. 启动服务

**启动 FastAPI 后端**（端口 8000）：

```bash
uv run uvicorn api.main:app --reload --port 8000
```

**启动 Next.js 前端**（端口 3000）：

```bash
cd web
npm run dev
```

打开浏览器访问 http://localhost:3000 即可使用 Web 界面。

### 4. 路线 C 额外要求（本地大模型）

Web 前端和 API 的 Embedding 使用 sentence-transformers (`bge-small-zh`)，首次运行会自动下载。  
LLM 生成需要 Ollama：

1. 安装 [Ollama](https://ollama.com) 并启动服务
2. 下载 LLM 模型：
   ```bash
   ollama pull qwen2.5:3b        # LLM 生成模型（~2GB）
   ```

> 命令行脚本 `lightrag_demo.py` 额外需要 `ollama pull nomic-embed-text`（~274MB）。

## Web 前端功能

前端使用 **Next.js + shadcn/ui + Tailwind CSS**，支持明暗主题切换。

| 页面 | 路径 | 功能 |
|------|------|------|
| 首页 | `/` | 项目介绍 + 三条路线导航 |
| 路线A | `/route-a` | BM25 关键词检索演示，得分柱状图 |
| 路线B | `/route-b` | 语义向量检索 + 混合检索，BM25 vs Embedding 对比 |
| 路线C | `/route-c` | 完整 RAG 流程动画：检索 → 构建 Prompt → LLM 流式生成 |

## 命令行脚本

除 Web 界面外，也可直接运行 Python 脚本：

### 🟢 路线A：BM25 关键词检索

```bash
uv run route_a_bm25/bm25_demo.py
```

中文分词 → BM25 打分 → Top-K 排序，交互式查询。

### 🟡 路线B：轻量深度学习

```bash
uv run route_b_embedding/embedding_demo.py    # 语义检索 Demo
uv run route_b_embedding/hybrid_search.py     # BM25 + Embedding 混合检索
uv run route_b_embedding/visualize_vectors.py # 生成向量空间可视化图表
```

### 🔴 路线C：完整 LLM RAG

```bash
uv run route_c_full_rag/simple_rag_demo.py    # 极简 RAG（从零手搓）
uv run route_c_full_rag/lightrag_demo.py      # LightRAG 知识图谱增强 RAG
uv run route_c_full_rag/visualize_pipeline.py # RAG 流程可视化图表
```

> ⚠️ 路线C需要 Ollama 运行中。首次运行 LightRAG 需索引文档（CPU 上约 10-15 分钟），后续运行直接加载缓存。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js 16 · TypeScript · Tailwind CSS · shadcn/ui · Framer Motion · Recharts |
| 后端 API | FastAPI · SSE (Server-Sent Events) · Uvicorn |
| 检索 | jieba 分词 · rank-bm25 · sentence-transformers (BGE-small-zh) |
| 生成 | Ollama (qwen2.5:3b) · LightRAG |
| 包管理 | uv (Python) · npm (Node.js) |

## 项目结构

```
├── web/                                # Next.js 前端
│   ├── src/app/                        # App Router 页面
│   │   ├── page.tsx                    #   首页
│   │   ├── route-a/page.tsx            #   路线A：BM25 演示
│   │   ├── route-b/page.tsx            #   路线B：Embedding 演示
│   │   └── route-c/page.tsx            #   路线C：完整 RAG 演示
│   ├── src/components/                 # 共享组件
│   └── src/lib/api.ts                  # API 客户端
│
├── api/                                # FastAPI 后端
│   ├── main.py                         #   入口 + CORS 配置
│   ├── routers/                        #   路由（bm25 / embedding / rag）
│   └── services/                       #   业务逻辑层
│
├── route_a_bm25/                       # 路线A 命令行脚本
│   └── bm25_demo.py
├── route_b_embedding/                  # 路线B 命令行脚本
│   ├── embedding_demo.py
│   ├── hybrid_search.py
│   ├── visualize_vectors.py
│   └── plots/                          # 生成的图表
├── route_c_full_rag/                   # 路线C 命令行脚本
│   ├── simple_rag_demo.py
│   ├── lightrag_demo.py
│   ├── visualize_pipeline.py
│   ├── plots/                          # 生成的图表
│   └── lightrag_data/                  # LightRAG 索引（自动生成）
│
├── sample_docs/
│   └── knowledge_base.json             # 10 篇中文知识库文档
├── docs/
│   └── issues-and-analysis.md          # 项目问题记录与分析
├── RAG技术路线选型指南.md
└── pyproject.toml
```
