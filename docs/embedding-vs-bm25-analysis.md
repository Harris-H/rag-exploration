# Embedding vs BM25 检索效果分析

## 问题描述

在 `simple_rag_demo.py` 中，对查询 **"LightRAG 和传统 RAG 有什么区别？"** 进行检索时：

| 检索方式 | Top-1 结果 | 得分 |
|---------|-----------|------|
| Embedding (nomic-embed-text) | Ollama 本地模型部署 ❌ | 0.5877 |
| BM25 关键词匹配 | LightRAG 框架介绍 ✅ | 8.2059 |

向量语义检索未能将最相关的文档排在首位，而 BM25 通过关键词精确匹配到了正确文档。

## 原因分析

### 1. 模型语言偏差

`nomic-embed-text` 是以英文语料为主训练的通用 Embedding 模型。对中文文本的语义理解能力有限，导致相似度得分区分度极低（Top-1 与 Top-2 仅差 **0.0008**）。

### 2. BM25 的关键词优势

查询中包含 "LightRAG" 这一专有名词，BM25 基于词频的精确匹配天然命中含该关键词的文档，得分差距显著（8.2 vs 5.1）。

### 3. 核心启示

- **单一检索方法各有盲区**：Embedding 擅长语义泛化（"怎么部署模型" ≈ "Ollama"），BM25 擅长精确关键词匹配
- **混合检索（Hybrid Search）是实践中的最佳方案**：用 RRF 等方法融合两种检索结果，取长补短
- **中文场景应选择中文专用模型**：如 `BAAI/bge-small-zh-v1.5`（Route B 使用）效果显著优于通用英文模型

## 改进方向

1. 将 Embedding 模型替换为中文专用模型（如 `bge-small-zh`）
2. 在检索阶段引入 RRF 混合排序（参考 `route_b_embedding/hybrid_search.py`）
