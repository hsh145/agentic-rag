# Agentic RAG — 多格式智能检索系统

基于 **LangGraph + FAISS + DashScope Embedding** 构建的 Agentic RAG（智能体检索增强生成）系统。  
支持 **PDF / Word / Excel / 图片 / 代码 / Markdown** 等多格式文件的自动解析、向量索引与智能检索。

---

## 架构

```
用户 Query
    │
    ▼
┌─────────────────────────────────────┐
│         LangGraph Agent              │
│                                      │
│  意图分析 → 文件解析 → 检索规划 →    │
│  执行检索 → 证据评估 → 生成回答       │
│         ↕ (条件边：不够就补搜)        │
└─────────────────────────────────────┘
    │
    ▼
  最终回答（带引用溯源）
```

## 核心能力

| 能力 | 说明 |
|------|------|
| **多格式解析** | PDF（PyMuPDF + camelot 表格提取）、Word、Excel、图片（OCR + VLM）、代码 |
| **语义分块** | Markdown 结构分块、代码分块、文本递归分块，表格/图片保持完整 |
| **向量检索** | DashScope text-embedding-v2 → FAISS 索引（1536 维） |
| **混合检索** | 向量语义检索 + BM25 关键词检索 + RRF 融合排序 |
| **Agent 闭环** | LangGraph 6 节点状态机，条件边实现"检索→评估→补搜"迭代 |
| **索引缓存** | FAISS 索引持久化，秒级启动 |
| **API 服务** | FastAPI + SSE 流式输出，开箱即用 |

## 快速开始

### 1. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

### 4. 调用 API

```bash
# 提问（不带文件）
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "SFT微调的关键参数有哪些"}'

# 提问（带文件解析）
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "总结这份文档的核心内容", "file_paths": ["/path/to/doc.pdf"]}'
```

## 项目结构

```
agentic-rag/
├── main.py              # FastAPI 入口
├── config.py            # 配置管理
├── agent/               # LangGraph Agent
│   ├── state.py         # 状态定义
│   ├── graph.py         # 工作流编排
│   ├── nodes.py         # 节点实现
│   └── tools.py         # 工具集
├── parser/              # 文件解析
│   ├── pdf_parser.py    # PDF + 表格
│   ├── office_parser.py # Word/Excel
│   ├── image_parser.py  # OCR + VLM
│   └── text_parser.py   # 文本/代码
└── rag/                 # RAG 核心
    ├── embedder.py      # Embedding
    ├── indexer.py       # FAISS 索引
    ├── retriever.py     # 混合检索
    └── chunker.py       # 语义分块
```

## 技术栈

- **Agent 框架**: LangGraph
- **向量检索**: FAISS + DashScope Embedding
- **混合检索**: BM25 + RRF 融合
- **文件解析**: PyMuPDF, camelot, python-docx, openpyxl, PaddleOCR
- **LLM**: 通义千问 (qwen-max/qwen-turbo)
- **Web 框架**: FastAPI + uvicorn

## 项目来源

本人在校项目，用于展示 AI 应用开发全栈能力。  
基于 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 的 RAG 核心思路重构并扩展为 Agentic 架构。
