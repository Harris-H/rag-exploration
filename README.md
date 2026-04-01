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
| 路线B | `/route-b` | 语义向量检索 + 混合检索 + Cross-Encoder 重排序 + 分块策略对比 + IR 指标评估 |
| 路线C | `/route-c` | 完整 RAG 管道：分块 → 混合检索 → 重排序 → Prompt → LLM 流式生成（增强/基线双模式） |

### 路线 B 详细功能（5 个 Tab）

| Tab | 功能 | 说明 |
|-----|------|------|
| 向量搜索 | Embedding 语义检索 | 使用 BGE-small-zh 模型，余弦相似度匹配 |
| 混合搜索 | BM25 + Embedding + RRF 融合 | 三列对比显示各方法结果 |
| 🔄 重排序 | Cross-Encoder 二次排序 | 对比重排前后的排名变化，含原理解释 |
| ✂️ 分块策略 | 4 种分块策略对比 | 固定长度/句子边界/滑动窗口/递归分块，可调分块大小 |
| 📏 检索评估 | MRR/P@K/R@K/NDCG 指标 | 10 条测试查询 × 3 种方法，柱状图 + 表格 + 逐查询详情 |

### 路线 C 详细功能（增强模式）

路线 C 提供 **增强模式** 和 **基线模式** 两种对比：

| 步骤 | 功能 | 说明 |
|------|------|------|
| 🔀 查询扩展 | LLM 生成查询变体 | 将原始问题改写为多个语义等价表述，多路召回 RRF 融合，可开关 |
| ✂️ 文档分块 | 4 种分块策略 | 递归(推荐)/句子边界/滑动窗口/固定长度，可调分块大小 50-500 字 |
| 🔍 混合检索 | BM25 + Embedding + RRF 融合 | 可切换纯向量/混合模式，支持多查询并行检索 |
| 🔄 重排序 | Cross-Encoder 精排 | 对比重排前后排名变化（↑/↓/═），可开关 |
| 📝 Prompt 构建 | 自动注入检索片段 | 带来源归属的参考片段拼接，指示 LLM 标注引用编号 |
| 🤖 LLM 流式生成 | Ollama 实时输出 | 逐 token 流式显示，SSE 推送 |
| 📎 引用标注 | 内联引用 + 来源列表 | 回答中 `[1]`、`[2]` 上标徽章，hover 显示来源文档，底部参考来源汇总 |
| 🔄 模型切换 | 多模型对比 | 支持在 qwen2.5:3b / qwen3.5:4b 等已安装模型间一键切换 |
| ⏱️ 阶段耗时 | Pipeline 性能分析 | 每个流水线步骤完成后显示耗时标签（绿色 <3s / 琥珀色 <10s / 红色 >10s），快速定位性能瓶颈 |

**引用标注机制**：Prompt 指示 LLM 在引用参考片段时标注来源编号 `[1]`、`[2]`；后端对 LLM 输出做后处理兜底（清除空括号、基于文本重叠自动注入遗漏的引用），确保小模型（3B）也能可靠输出引用。前端 `CitedText` 组件将 `[N]` 渲染为翠绿色上标圆形徽章，悬浮显示来源文档标题。

**阶段耗时机制**：后端在每个 SSE 事件中携带 `elapsed_ms` 字段，前端 `PipelineStep` 组件在步骤完成时渲染色彩编码标签——绿色（<3 秒）表示正常，琥珀色（3-10 秒）需关注，红色（>10 秒）为瓶颈。生成阶段耗时从 `done` 事件的 `generation_ms` 字段获取。

**左侧** 实时 Pipeline 状态可视化（7 步流程动画 + 耗时标签），**右侧** 逐步展示查询扩展变体、分块信息、检索结果、重排序对比、Prompt 和流式回答。

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
uv run route_b_embedding/reranking_demo.py    # Cross-Encoder 重排序
uv run route_b_embedding/visualize_vectors.py # 生成向量空间可视化图表
```

### 📏 评估 & 分块

```bash
uv run evaluation/eval_demo.py                # IR 指标评估（MRR/NDCG/P@K/R@K）
uv run preprocessing/chunking_demo.py         # 4 种分块策略对比
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
| 检索 | jieba 分词 · rank-bm25 · sentence-transformers (BGE-small-zh) · Cross-Encoder (BGE-reranker-base) |
| 生成 | Ollama (qwen2.5:3b) · LightRAG |
| 包管理 | uv (Python) · npm (Node.js) |

## 项目结构

```
├── web/                                # Next.js 前端
│   ├── src/app/                        # App Router 页面
│   │   ├── page.tsx                    #   首页
│   │   ├── route-a/page.tsx            #   路线A：BM25 演示
│   │   ├── route-b/                    #   路线B：Embedding 演示
│   │   │   ├── page.tsx                #     主页面（5 个 Tab）
│   │   │   ├── RerankingTab.tsx        #     重排序 Tab 组件
│   │   │   ├── ChunkingTab.tsx         #     分块策略 Tab 组件
│   │   │   └── EvaluationTab.tsx       #     检索评估 Tab 组件
│   │   └── route-c/page.tsx            #   路线C：完整 RAG 演示
│   ├── src/components/                 # 共享组件（含 CitedText 引用渲染）
│   └── src/lib/api.ts                  # API 客户端
│
├── api/                                # FastAPI 后端
│   ├── main.py                         #   入口 + CORS 配置
│   ├── routers/                        #   路由（bm25 / embedding / rag / enhanced_rag / reranking / chunking / eval）
│   └── services/                       #   业务逻辑层（含 enhanced_rag_service）
│
├── route_a_bm25/                       # 路线A 命令行脚本
│   └── bm25_demo.py
├── route_b_embedding/                  # 路线B 命令行脚本
│   ├── embedding_demo.py
│   ├── hybrid_search.py
│   ├── reranking_demo.py               # Cross-Encoder 重排序
│   ├── visualize_vectors.py
│   └── plots/                          # 生成的图表
├── route_c_full_rag/                   # 路线C 命令行脚本
│   ├── simple_rag_demo.py
│   ├── lightrag_demo.py
│   ├── visualize_pipeline.py
│   ├── plots/                          # 生成的图表
│   └── lightrag_data/                  # LightRAG 索引（自动生成）
│
├── evaluation/                         # 检索评估
│   └── eval_demo.py                    # MRR/NDCG/P@K/R@K 评估脚本
├── preprocessing/                      # 预处理工具
│   └── chunking_demo.py                # 4 种分块策略对比
│
├── sample_docs/
│   └── knowledge_base.json             # 15 篇中文知识库文档
├── docs/
│   └── issues-and-analysis.md          # 项目问题记录与分析
├── RAG技术路线选型指南.md
└── pyproject.toml
```

## 模型下载说明

项目启动时默认 `HF_HUB_OFFLINE=1`，需提前下载模型。中国大陆用户可使用镜像：

```powershell
$env:HF_HUB_OFFLINE="0"
$env:HF_ENDPOINT="https://hf-mirror.com"
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"
```

| 模型 | 用途 | 大小 |
|------|------|------|
| BAAI/bge-small-zh-v1.5 | Embedding 向量化（512 维） | ~90MB |
| BAAI/bge-reranker-base | Cross-Encoder 重排序 | ~400MB |
| qwen2.5:3b (Ollama) | 路线C LLM 生成 | ~2GB |
