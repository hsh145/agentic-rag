"""
将 CRUD-RAG 新闻文档导入 FAISS 索引

用法：
    python scripts/index_crud_docs.py

效果：
    将 questanswer 任务的新闻文档索引到 data/index 目录，
    然后 /api/ask 就能检索到这些文档内容。
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import EmbeddingManager
from rag.indexer import IndexManager
from rag import get_chunker
from langchain_core.documents import Document
from config import DEFAULT_CONFIG

DOCS_PATH = "data/eval/crud_rag/split_merged.json"

# 1. 提取所有新闻文档
print("Loading CRUD-RAG news documents...")
with open(DOCS_PATH, encoding="utf-8") as f:
    raw = json.load(f)

documents = []
for task in ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]:
    for item in raw.get(task, []):
        for key in ["news1", "news2", "news3"]:
            text = item.get(key, "")
            if text and len(text) > 50:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": f"crud_rag_{task}",
                        "source_type": "text",
                        "doc_id": item.get("ID", ""),
                        "task_type": task,
                    }
                ))

print(f"  Extracted {len(documents)} documents")
print(f"  Total chars: {sum(len(d.page_content) for d in documents):,}")

# 2. 分块
print("\nChunking documents...")
chunker = get_chunker()()
chunks = chunker.chunk_all(documents)
print(f"  -> {len(chunks)} chunks")

# 3. 建立索引
print(f"\nBuilding FAISS index (model: {DEFAULT_CONFIG.embedding_model})...")
t0 = time.time()

from dotenv import load_dotenv
load_dotenv()

embedder = EmbeddingManager(DEFAULT_CONFIG.embedding_model)
indexer = IndexManager(embedder.get_embeddings(), DEFAULT_CONFIG.index_save_path)

indexer.build_index(chunks)
indexer.save_index()

elapsed = time.time() - t0
print(f"\nDone! {elapsed:.1f}s")
print(f"  Total vectors: {indexer.index.ntotal}")
print(f"  Index saved to: {DEFAULT_CONFIG.index_save_path}")
print(f"\nNow you can run CRUD-RAG evaluation.")
