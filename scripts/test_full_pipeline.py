"""
完整流水线测试：解析 → 分块 → 建索引 → 问答
测试所有文件格式（PDF/Word/Excel/图片）

用法:
    python scripts/test_full_pipeline.py
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agent.tools import DocumentParserTool
from rag import get_chunker
from rag.embedder import EmbeddingManager
from rag.indexer import IndexManager
from langchain_core.documents import Document
from config import DEFAULT_CONFIG


def test_pipeline():
    print("=" * 55)
    print("  Full Pipeline Test: Parse → Chunk → Index → Query")
    print("=" * 55)
    print()

    # ========== 1. 解析所有 benchmark 文件 ==========
    print("[1/4] Parsing files...")
    parser = DocumentParserTool()

    test_files = [
        ("data/benchmark/benchmark.pdf", "PDF"),
        ("data/benchmark/benchmark.docx", "Word"),
        ("data/benchmark/benchmark.xlsx", "Excel"),
        ("data/benchmark/benchmark_ocr.png", "Image"),
    ]

    all_docs = []
    for file_path, ftype in test_files:
        t0 = time.time()
        docs, errors = parser.parse_files([file_path])
        elapsed = time.time() - t0
        if docs:
            chars = len(docs[0].page_content)
            print(f"  [{ftype:>5}] {os.path.basename(file_path):>20} -> {chars:>5} chars, {elapsed:.2f}s")
            all_docs.extend(docs)
        else:
            print(f"  [{ftype:>5}] {os.path.basename(file_path):>20} -> FAILED: {errors}")

    print(f"  Total: {len(all_docs)} documents")
    print()

    # ========== 2. 分块 ==========
    print("[2/4] Chunking...")
    chunker_cls = get_chunker()
    chunker = chunker_cls()
    chunks = chunker.chunk_all(all_docs)
    print(f"  -> {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"  Chunk {i}: [{c.metadata.get('chunk_id','?')}] {c.page_content[:80]}...")
    print()

    # ========== 3. 建索引 ==========
    print("[3/4] Building index...")
    embedder = EmbeddingManager(DEFAULT_CONFIG.embedding_model)
    indexer = IndexManager(embedder.get_embeddings(), "./data/tmp_index")
    t0 = time.time()
    indexer.build_index(chunks)
    indexer.save_index()
    elapsed = time.time() - t0
    print(f"  -> {indexer.index.ntotal} vectors, {elapsed:.1f}s")
    print()

    # ========== 4. 问答 ==========
    print("[4/4] Asking questions based on parsed docs...")
    queries = [
        "系统在哪些评估指标上表现如何？具体数字是多少？",
        "模型配置对比表中，Qwen-7B的参数量和显存占用是多少？",
        "季度销售数据中，C产品Q2的销售额是多少？增长率如何？",
    ]

    from rag.retriever import HybridRetriever
    retriever = HybridRetriever(indexer, chunks)

    for qi, query in enumerate(queries):
        print(f"\n  Q{qi+1}: {query}")
        t0 = time.time()

        # 检索
        results = retriever.hybrid_search(query, top_k=3)

        if not results:
            print(f"  -> No results found!")
            continue

        # 用 LLM 生成回答
        from agent.nodes import _get_llm_for_generate
        from langchain_core.messages import HumanMessage

        context = "\n---\n".join([d.page_content[:800] for d in results])
        llm = _get_llm_for_generate()
        prompt = f"""基于以下检索到的信息回答问题。

问题：{query}

参考信息：
{context[:5000]}

回答要求：基于参考信息回答，给出具体数字和数据。"""
        resp = llm.invoke([HumanMessage(content=prompt)])
        elapsed = time.time() - t0
        print(f"  -> {resp.content[:300]}")
        print(f"     ({elapsed:.1f}s, {len(results)} chunks)")


if __name__ == "__main__":
    test_pipeline()
