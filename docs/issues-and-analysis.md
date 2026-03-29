# 项目踩坑记录与问题分析

本文档记录 rag-exploration 项目开发过程中遇到的问题、原因分析及解决方案。

---

## 1. Embedding 模型中文检索效果不如 BM25

**场景：** `simple_rag_demo.py` 查询 "LightRAG 和传统 RAG 有什么区别？"

| 检索方式 | Top-1 结果 | 得分 |
|---------|-----------|------|
| Embedding (nomic-embed-text) | Ollama 本地模型部署 ❌ | 0.5877 |
| BM25 关键词匹配 | LightRAG 框架介绍 ✅ | 8.2059 |

**原因：**
- `nomic-embed-text` 以英文语料训练，中文语义区分度极低（Top-1 与 Top-2 仅差 0.0008）
- BM25 对精确关键词 "LightRAG" 天然命中，得分差距显著
- **启示：** 单一检索有盲区，中文场景应选中文模型（如 `bge-small-zh`）或使用混合检索

**改进方向：** 替换为中文 Embedding 模型，或引入 RRF 混合排序（参考 `route_b_embedding/hybrid_search.py`）

---

## 2. 系统代理导致 Ollama 请求 502 Bad Gateway

**场景：** 系统设置了 `HTTP_PROXY=http://127.0.0.1:7897`，Python httpx 请求 `localhost:11434` 被代理拦截

**原因：** 系统代理未排除 localhost，所有 HTTP 请求（包括本地）都经过代理转发，代理无法连接 Ollama 返回 502

**解决：** 在脚本最顶部（import httpx 之前）设置环境变量：
```python
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1"
```

---

## 3. LightRAG Embedding 维度不匹配

**场景：** 使用 `nomic-embed-text`（768 维）配合 LightRAG 内置的 `ollama_embed` 函数报维度错误

**原因：** LightRAG 的 `ollama_embed` 函数有硬编码装饰器 `@wrap_embedding_func_with_attrs(embedding_dim=1024)`（针对 bge-m3），会覆盖用户传入的 `embedding_dim=768` 配置

**解决：** 不使用 LightRAG 内置的 `ollama_embed`，改写自定义 Embedding 函数直接调用 `ollama.AsyncClient.embed()`

---

## 4. LLM 推理超时（CPU 环境）

**场景：** LightRAG 在 CPU 推理时 Worker 超时退出

**原因：** LightRAG 默认 `DEFAULT_LLM_TIMEOUT = 180s`，Worker 超时为 2 倍 = 360s。CPU 推理 qwen2.5:3b 单次生成可能超过 360s

**解决：** 构造函数传入 `default_llm_timeout=600`（Worker 超时自动变为 1200s），同时 Ollama kwargs 设置 `"timeout": 0`（无限等待）

---

## 5. Anaconda Python 与 PyTorch/ONNX DLL 冲突

**场景：** 使用 Anaconda Python 3.11 运行 `sentence-transformers` 时出现 DLL 加载失败

**原因：** Anaconda 自带的 DLL 依赖与 PyTorch CPU 版和 onnxruntime 的 DLL 产生冲突（Windows 特有问题）

**解决：** 切换为独立安装的 Python 3.11.5（`D:\Language\Python\Python311\python.exe`），用 `uv venv --python` 指定路径重建虚拟环境

---

## 6. LightRAG Storage 初始化遗漏

**场景：** 构造 `LightRAG()` 后直接调用 `insert()` 报 `StorageNotInitializedError`

**原因：** LightRAG 构造函数不会自动初始化存储后端，需显式调用

**解决：** 构造后加 `await rag.initialize_storages()`

---

## 7. Ollama 服务启动问题

**场景：** 通过 "Ollama app.exe" 托盘启动后 API 无响应

**原因：** GUI 版本有时不能正确启动内部 serve 进程

**解决：** 直接运行 `ollama.exe serve` 启动服务，更可靠
