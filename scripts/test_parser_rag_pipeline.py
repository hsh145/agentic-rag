"""
端到端测试：PDF/Word/Excel 解析 -> 分块 -> Embedding -> RAG 检索 -> QA 测试
"""
import sys, os, json, time, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 .env 中的 API key
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)

from config import DEFAULT_CONFIG
from parser.pdf_parser import PDFParser
from parser.office_parser import OfficeParser
from parser.text_parser import TextParser
from parser.detector import FileTypeDetector
from rag import get_chunker
from rag.embedder import EmbeddingManager
from rag.indexer import IndexManager
from rag.retriever import HybridRetriever
from langchain_core.documents import Document

BENCHMARK_DIR = Path("data/benchmark")
INDEX_DIR = Path("data/tmp_index")

results = {
    "parsing": {},
    "chunking": {},
    "embedding": {},
    "retrieval": {},
    "qa": {},
}

# ================================================================
# 1. 解析测试
# ================================================================
print("=" * 60)
print("  阶段1: 文档解析 (PDF/Word/Excel)")
print("=" * 60)

test_files = [
    ("benchmark.pdf", PDFParser(), "pdf"),
    ("benchmark.docx", OfficeParser(), "docx"),
    ("benchmark.xlsx", OfficeParser(), "xlsx"),
]

parsed_docs = []

for fname, parser, ftype in test_files:
    fpath = str(BENCHMARK_DIR / fname)
    t0 = time.time()
    try:
        docs = parser.parse(fpath)
        elapsed = time.time() - t0
        total_chars = sum(len(d.page_content) for d in docs)
        results["parsing"][fname] = {
            "status": "ok",
            "time_s": round(elapsed, 3),
            "num_docs": len(docs),
            "total_chars": total_chars,
        }
        print(f"  [OK] {fname:20s} {len(docs):2d} docs, {total_chars:>5} chars, {elapsed:.3f}s")
        parsed_docs.extend(docs)
    except Exception as e:
        elapsed = time.time() - t0
        results["parsing"][fname] = {"status": "fail", "error": str(e)}
        print(f"  [FAIL] {fname}: {e}")

# 添加 sample.txt 作为纯文本参考
txt_parser = TextParser()
for fname in ["sample.txt", "sample.md", "sample.json"]:
    fpath = str(BENCHMARK_DIR / fname)
    t0 = time.time()
    try:
        docs = txt_parser.parse(fpath)
        elapsed = time.time() - t0
        total_chars = sum(len(d.page_content) for d in docs)
        results["parsing"][fname] = {
            "status": "ok",
            "time_s": round(elapsed, 3),
            "num_docs": len(docs),
            "total_chars": total_chars,
        }
        print(f"  [OK] {fname:20s} {len(docs):2d} docs, {total_chars:>5} chars, {elapsed:.3f}s")
        parsed_docs.extend(docs)
    except Exception as e:
        elapsed = time.time() - t0
        results["parsing"][fname] = {"status": "fail", "error": str(e)}
        print(f"  [FAIL] {fname}: {e}")

# ================================================================
# 2. 分块
# ================================================================
print(f"\n{'=' * 60}")
print(f"  阶段2: 语义分块")
print(f"=" * 60)

t0 = time.time()
chunker_cls = get_chunker()
chunker = chunker_cls()
chunks = chunker.chunk_all(parsed_docs)
elapsed = time.time() - t0

results["chunking"] = {
    "input_docs": len(parsed_docs),
    "output_chunks": len(chunks),
    "time_s": round(elapsed, 3),
    "chunk_sizes": {
        "min": min(len(c.page_content) for c in chunks) if chunks else 0,
        "max": max(len(c.page_content) for c in chunks) if chunks else 0,
        "avg": round(sum(len(c.page_content) for c in chunks) / len(chunks)) if chunks else 0,
    },
}
print(f"  {len(parsed_docs)} docs -> {len(chunks)} chunks ({elapsed:.3f}s)")
print(f"  Chunk size: min={results['chunking']['chunk_sizes']['min']} max={results['chunking']['chunk_sizes']['max']} avg={results['chunking']['chunk_sizes']['avg']}")

# 打印每个 chunk 的 source 分布
source_types = {}
for c in chunks:
    st = c.metadata.get("source_type", "unknown")
    source_types[st] = source_types.get(st, 0) + 1
print(f"  Chunk source types: {source_types}")

# ================================================================
# 3. Embedding + 索引
# ================================================================
print(f"\n{'=' * 60}")
print(f"  阶段3: Embedding + FAISS 索引")
print(f"=" * 60)

t0 = time.time()
embedder = EmbeddingManager(DEFAULT_CONFIG.embedding_model)
indexer = IndexManager(embedder.get_embeddings(), str(INDEX_DIR))

# 读取已有索引
has_existing = indexer.load_index()

if has_existing:
    print(f"  已有索引: {indexer.index.ntotal} 个向量")
    # 追加新文档
    new_chunks = [c for c in chunks if c.page_content not in set(d.page_content for d in indexer.documents)]
    if new_chunks:
        print(f"  追加 {len(new_chunks)} 个新块...")
        indexer.add_documents(new_chunks)
        indexer.save_index()
    embed_time = time.time() - t0
    total_vectors = indexer.index.ntotal
    print(f"  索引更新完成: {total_vectors} 个总向量 ({embed_time:.3f}s)")
else:
    print(f"  新建索引...")
    indexer.build_index(chunks)
    indexer.save_index()
    embed_time = time.time() - t0
    total_vectors = indexer.index.ntotal
    print(f"  索引构建完成: {total_vectors} 个向量 ({embed_time:.3f}s)")

results["embedding"] = {
    "model": DEFAULT_CONFIG.embedding_model,
    "total_vectors": total_vectors,
    "time_s": round(embed_time, 3),
}

# ================================================================
# 4. 检索测试
# ================================================================
print(f"\n{'=' * 60}")
print(f"  阶段4: RAG 混合检索测试")
print(f"=" * 60)

all_chunks = getattr(indexer, "documents", [])
retriever = HybridRetriever(indexer, all_chunks, rrf_k=DEFAULT_CONFIG.rrf_k)

test_queries = [
    "LoRA的rank和alpha参数推荐值是多少？",
    "Qwen-7B和Qwen-14B的MMLU分数分别是多少？",
    "A产品在Q1和Q2的销售额分别是多少？",
    "什么是SFT微调？",
    "模型量化可以降低多少显存占用？",
]

retrieval_results = []
for q in test_queries:
    t0 = time.time()
    results_q = retriever.hybrid_search(q, top_k=DEFAULT_CONFIG.top_k)
    elapsed = time.time() - t0
    retrieval_results.append((q, results_q, elapsed))

    print(f"\n  Query: {q}")
    print(f"  Time: {elapsed:.4f}s, Results: {len(results_q)}")
    for j, r in enumerate(results_q[:3]):
        content = r.page_content[:150].replace("\n", " ")
        score = r.metadata.get("rrf_score", r.metadata.get("similarity_score", "?"))
        src = r.metadata.get("source_type", r.metadata.get("file_name", "?"))
        print(f"    [{j+1}] score={score:.3f} src={src}")
        print(f"         {content}...")

results["retrieval"] = {
    "num_queries": len(test_queries),
    "avg_time_s": round(sum(r[2] for r in retrieval_results) / len(retrieval_results), 4),
    "results": [{"query": q, "time": t, "num_results": len(r)} for q, r, t in retrieval_results],
}

# ================================================================
# 5. QA 测试（用 LLM 生成回答）
# ================================================================
print(f"\n{'=' * 60}")
print(f"  阶段5: RAG QA 端到端测试")
print(f"=" * 60)

qa_results = []

qa_pairs = [
    {
        "question": "LoRA的rank和alpha参数推荐值是多少？",
        "expected_topics": ["rank=8", "alpha=16"],
    },
    {
        "question": "Qwen-7B和Qwen-14B在C-Eval上的分数分别是多少？",
        "expected_topics": ["72.3", "78.2"],
    },
    {
        "question": "A产品Q1和Q2的销售额及同比增长率是多少？",
        "expected_topics": ["1280", "1450", "12.5%", "15.2%"],
    },
]

for qa in qa_pairs:
    q = qa["question"]
    print(f"\n  Q: {q}")

    t0 = time.time()
    try:
        # 直接用混合检索器
        retrieved = retriever.hybrid_search(q, top_k=3)
        elapsed = time.time() - t0

        if retrieved:
            all_text = " ".join(r.page_content for r in retrieved)
            found_topics = [t for t in qa["expected_topics"] if t in all_text]
            missing_topics = [t for t in qa["expected_topics"] if t not in all_text]

            print(f"  Time: {elapsed:.3f}s")
            print(f"  Retrieved: {len(retrieved)} chunks")
            print(f"  Topics found: {found_topics}")
            if missing_topics:
                print(f"  Topics MISSING: {missing_topics}")

            qa_results.append({
                "question": q,
                "success": len(missing_topics) == 0,
                "time_s": round(elapsed, 3),
                "retrieved_count": len(retrieved),
                "found_topics": found_topics,
                "missing_topics": missing_topics,
            })
        else:
            print(f"  Time: {elapsed:.3f}s")
            print(f"  NO results retrieved!")
            qa_results.append({
                "question": q,
                "success": False,
                "time_s": round(elapsed, 3),
                "error": "no results",
            })
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR: {e}")
        qa_results.append({
            "question": q,
            "success": False,
            "time_s": round(elapsed, 3),
            "error": str(e),
        })

results["qa"] = {
    "total": len(qa_results),
    "success": sum(1 for r in qa_results if r["success"]),
    "failed": sum(1 for r in qa_results if not r["success"]),
    "avg_time_s": round(sum(r["time_s"] for r in qa_results) / len(qa_results), 3),
    "details": qa_results,
}

# ================================================================
# 报告汇总
# ================================================================
print(f"\n{'=' * 60}")
print(f"  测试报告汇总")
print(f"{'=' * 60}")

print(f"""
1. 文档解析
   - PDF (benchmark.pdf): {results['parsing']['benchmark.pdf']['status']}, {results['parsing']['benchmark.pdf'].get('total_chars',0)} chars, {results['parsing']['benchmark.pdf'].get('time_s',0):.3f}s
   - Word (benchmark.docx): {results['parsing']['benchmark.docx']['status']}, {results['parsing']['benchmark.docx'].get('total_chars',0)} chars, {results['parsing']['benchmark.docx'].get('time_s',0):.3f}s
   - Excel (benchmark.xlsx): {results['parsing']['benchmark.xlsx']['status']}, {results['parsing']['benchmark.xlsx'].get('total_chars',0)} chars, {results['parsing']['benchmark.xlsx'].get('time_s',0):.3f}s
   - TXT/MD/JSON: included

2. 分块: {results['chunking']['input_docs']} docs -> {results['chunking']['output_chunks']} chunks ({results['chunking']['time_s']:.3f}s)
   - chunk size: {results['chunking']['chunk_sizes']['avg']} avg, {results['chunking']['chunk_sizes']['max']} max

3. Embedding: {results['embedding']['total_vectors']} vectors, {results['embedding']['time_s']:.3f}s

4. 检索测试: {results['retrieval']['num_queries']} queries, avg {results['retrieval']['avg_time_s']:.4f}s/query

5. QA 测试: {results['qa']['success']}/{results['qa']['total']} passed, avg {results['qa']['avg_time_s']:.3f}s/query
""")

# 保存详细报告
report_path = BENCHMARK_DIR.parent / "eval" / "pipeline_test_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"  详细报告已保存: {report_path}")
