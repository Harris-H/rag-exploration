# RAG 技术探索 —— 从 BM25 到完整 LLM RAG

可视化理解 RAG（检索增强生成）检索流程的学习项目。

## 环境准备

```bash
# 需要 Python 3.11+, 推荐使用 uv 包管理器
uv sync
```

> 💡 首次运行路线B时会自动下载 Embedding 模型（约 50MB）。  
> 中国大陆用户如下载缓慢，可设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`

## 🟢 路线A：BM25 关键词检索

纯传统方法，无需 GPU，毫秒级响应。

```bash
uv run route_a_bm25/bm25_demo.py
```

展示内容：
- 中文分词 → BM25 打分 → Top-K 排序
- 可视化分数条和排名表
- 交互式查询（支持示例快捷选择）

## 🟡 路线B：轻量深度学习

使用 BGE-small-zh 神经网络 Embedding 模型，CPU 可运行。

### 语义检索 Demo

```bash
uv run route_b_embedding/embedding_demo.py
```

对比 BM25（词级匹配）vs Embedding（语义匹配），展示同义词理解能力。

### 混合检索 Demo

```bash
uv run route_b_embedding/hybrid_search.py
```

BM25 + Embedding 融合（RRF 算法），三路结果对比 + 权重敏感性分析。

### 向量空间可视化

```bash
uv run route_b_embedding/visualize_vectors.py
```

生成 PNG 图表到 `route_b_embedding/plots/`：
- PCA 二维散点图（文档分布 + 查询投影）
- 语义相似度热力图
- 查询相似度柱状图

## 🔴 路线C：完整 LLM RAG

使用 Ollama 本地大模型 + LightRAG 知识图谱框架，实现完整的「检索 → 生成」流程。

### 前置要求

1. 安装 [Ollama](https://ollama.com) 并启动服务
2. 下载模型：
   ```bash
   ollama pull qwen2.5:3b        # LLM 生成模型（~2GB）
   ollama pull nomic-embed-text   # Embedding 模型（~274MB）
   ```

### 极简 RAG Demo（从零手搓）

```bash
uv run route_c_full_rag/simple_rag_demo.py
```

不依赖任何框架，展示 RAG 本质：检索文档 → 拼接 Prompt → 调用 LLM 生成。  
对比有 RAG vs 无 RAG 的回答质量。

### LightRAG 完整演示

```bash
uv run route_c_full_rag/lightrag_demo.py
```

LightRAG 知识图谱增强 RAG：
- 自动抽取实体和关系，构建知识图谱
- 五种检索模式对比：naive / local / global / hybrid / mix
- 增量更新演示（新增文档无需重建索引）

> ⚠️ 首次运行需索引文档（LLM 在 CPU 上约 10-15 分钟），后续运行直接加载缓存。

### RAG 流程可视化

```bash
uv run route_c_full_rag/visualize_pipeline.py
```

生成 PNG 图表到 `route_c_full_rag/plots/`：
- RAG 全流程架构图
- 检索相关度分布
- 查询-文档相似度热力图
- RAG vs 无 RAG 回答对比

## 项目结构

```
├── route_a_bm25/
│   └── bm25_demo.py                # BM25 检索演示
├── route_b_embedding/
│   ├── embedding_demo.py           # 语义检索演示
│   ├── hybrid_search.py            # 混合检索演示
│   ├── visualize_vectors.py        # 向量空间可视化
│   └── plots/                      # 生成的图表
├── route_c_full_rag/
│   ├── simple_rag_demo.py          # 极简 RAG（手搓版）
│   ├── lightrag_demo.py            # LightRAG 完整演示
│   ├── visualize_pipeline.py       # RAG 流程可视化
│   ├── plots/                      # 生成的图表
│   └── lightrag_data/              # LightRAG 索引数据（自动生成）
├── sample_docs/
│   └── knowledge_base.json         # 10 篇中文知识库文档
├── RAG技术路线选型指南.md            # 完整技术路线分析
└── pyproject.toml
```
