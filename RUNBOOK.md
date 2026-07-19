# Agentic RAG 运维手册

## 1. 服务无法启动

**现象**: `uvicorn main:app` 启动后立即退出

**排查**:

```bash
# 1. 检查端口占用
netstat -ano | findstr :8000

# 2. 检查 .env 配置
cat .env | grep DASHSCOPE_API_KEY

# 3. 检查依赖
python -c "import fastapi; import uvicorn; print('OK')"
```

**恢复**:
```bash
# 端口被占用
netstat -ano | findstr :8000 | findstr LISTEN
taskkill /f /pid <PID>

# API Key 缺失 → 补充 .env
echo DASHSCOPE_API_KEY=sk-xxx >> .env
```

---

## 2. FAISS 索引损坏

**现象**: `IndexManager.load_index()` 返回 False 或报错

**恢复**:

```bash
# 方案A: 重建索引（需要 .env 配置 API Key）
python -c "
from rag.embedder import EmbeddingManager
from rag.indexer import IndexManager
from config import DEFAULT_CONFIG as cfg
from parser.detector import FileTypeDetector
from agent.tools import DocumentParserTool
from rag import get_chunker
from langchain_core.documents import Document as LCDocument
from pathlib import Path

parser = DocumentParserTool()
docs, _ = parser.parse_files(['./data/benchmark/sample.txt', './data/benchmark/sample.md'])
chunker_cls = get_chunker()
chunks = chunker_cls().chunk_all(docs)
embedder = EmbeddingManager(cfg.embedding_model)
indexer = IndexManager(embedder.get_embeddings(), cfg.index_save_path)
indexer.build_index(chunks)
indexer.save_index()
print('修复完成')
"

# 方案B: 直接删除索引目录重新构建
rm -rf data/index/
# 然后重启服务，首次请求会自动建索引
```

---

## 3. 长期记忆损坏 / 脏数据

**现象**: 回答中包含过时或矛盾的事实

**恢复**:

```bash
# 1. 查看当前事实
python -c "
import sqlite3
conn = sqlite3.connect('data/memory.db')
rows = conn.execute('SELECT id, fact_text, entity, created_at FROM facts ORDER BY id DESC LIMIT 20').fetchall()
for r in rows: print(r)
"

# 2. 清理旧事实（保留最近 30 天）
python -c "
from agent.memory import AgentMemory
m = AgentMemory()
deleted = m.forget_old_facts(30)
print(f'删除了 {deleted} 条旧事实')
"

# 3. 或通过 API
curl -X POST http://localhost:8000/api/memory/forget \
  -H "Content-Type: application/json" \
  -d '{"strategy": "old", "days": 30}'
```

---

## 4. 卡死在检索循环

**现象**: 一个请求跑了超过 5 分钟不返回

**排查**:
1. 查看日志找 `iteration` 值
2. 检查 `evaluate_evidence` 是否每次都返回 `needs_more=True`

**恢复**:

```bash
# 检查请求状态
tail -50 logs/rag_*.log | grep -E "iteration|needs_more|evaluate"

# 临时降低迭代上限
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "测试", "max_iterations": 1}' \
  --max-time 30
```

**代码修复**: `agent/nodes.py` 中 `evaluate_evidence` 的 LLM fallback 回退可能不够敏感，
如果 `LLM 评估失败` 走到规则回退，确保 `can_answer` 判断合理。

---

## 5. Embedding API 限流 / 超时

**现象**: 日志中出现 `Embedding API 失败` 或 `401`

**处理**:

```bash
# 1. 检查 API Key 是否过期
curl -X POST https://dashscope.aliyun.com/api/v1/services/embeddings/text-embedding \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-v2", "input": {"texts": ["test"]}}'

# 2. 如果在 embedder.py 中启用了 batch，降低 batch 大小
# embedder.py: batch_size=16 → batch_size=8
```

---

## 6. 回答质量突然下降

**排查链条**:

```bash
# 1. 跑 golden set 看 pass rate
python tests/eval_golden.py

# 2. 跑检索评估看 recall
python tests/eval_retrieval.py

# 3. 检查 memory 是否被脏数据污染
python -c "
import sqlite3
conn = sqlite3.connect('data/memory.db')
count = conn.execute('SELECT COUNT(*) FROM facts').fetchone()
print(f'{count[0]} 条事实')
print('最近 5 条:')
for r in conn.execute('SELECT id, fact_text FROM facts ORDER BY id DESC LIMIT 5').fetchall():
    print(f'  [{r[0]}] {r[1][:60]}')
"
```

**恢复**:
```bash
# 清理低置信度事实
curl -X POST http://localhost:8000/api/memory/forget \
  -H "Content-Type: application/json" \
  -d '{"strategy": "low_confidence", "threshold": 0.3}'
```

---

## 7. Streamlit 前端白屏

**现象**: 打开 http://localhost:8501 只看到空白

**排查**:

```bash
# 1. 检查前端进程
tasklist | grep streamlit

# 2. 检查后端是否可访问
curl http://localhost:8000/api/health

# 3. 重启前端
taskkill /f /im streamlit 2>/dev/null
python -m streamlit run web/app.py --server.port 8501 --server.headless true
```

---

## 8. 数据库迁移 / 重建

```bash
# 备份
copy data/memory.db data/memory.db.bak

# 完全重置（小心：会丢失所有记忆）
python -c "
from agent.memory import AgentMemory
m = AgentMemory()
# 删除 facts 表重建
import sqlite3
conn = sqlite3.connect('data/memory.db')
conn.executescript('DROP TABLE IF EXISTS facts;')
conn.close()
print('facts 表已重置')
"
```

---

## 9. 日志查看

```bash
# 实时日志
tail -f logs/rag_$(date +%Y-%m-%d).log

# 只看错误
grep -i "error\|exception\|traceback" logs/rag_*.log

# 按 session 筛选
grep "session_abc123" logs/rag_*.log
```
