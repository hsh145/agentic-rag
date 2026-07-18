"""
检索层评估 — Recall@k / Precision@k / MRR / nDCG@k

独立于 LLM 的纯检索评估，衡量"搜得对不对"。
这是所有 RAG 指标的地基，跑得最快、最便宜、最可重复。

用法：
    python tests/eval_retrieval.py

输出：
    - 控制台报告（含总体 + 按难度分层 + 每条详情）
    - data/eval/retrieval_report.json（结构化结果）
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 加载 .env（main.py 在启动时做，eval 脚本需要手动加载）
_env_file = ROOT / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("DASHSCOPE_API_KEY="):
                _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                if _val:
                    os.environ["DASHSCOPE_API_KEY"] = _val
            elif _line.startswith("MOONSHOT_API_KEY="):
                _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                if _val:
                    os.environ["MOONSHOT_API_KEY"] = _val


# ============================================================
# 1. 索引构建：解析 benchmark 文档 → 分块 → FAISS + BM25
# ============================================================

def build_benchmark_index():
    """解析 data/benchmark/ 下的所有文档，构建检索索引

    Returns:
        (retriever, chunk_index)
        chunk_index: List[Dict]，每项 {"chunk_id", "content", "source_file", "chunk"}
    """
    from parser.detector import FileTypeDetector
    from agent.tools import DocumentParserTool
    from rag import get_chunker
    from rag.embedder import EmbeddingManager
    from rag.indexer import IndexManager
    from rag.retriever import HybridRetriever
    from langchain_core.documents import Document as LCDocument
    from config import DEFAULT_CONFIG

    benchmark_dir = ROOT / "data" / "benchmark"
    if not benchmark_dir.exists():
        print("❌ data/benchmark/ 目录不存在")
        sys.exit(1)

    # 收集所有文档
    doc_files = sorted(benchmark_dir.iterdir())
    file_paths = [str(f) for f in doc_files if f.is_file() and f.suffix != ".json"]
    # manifest.json 不是文档，排除
    file_paths = [p for p in file_paths if "manifest" not in p]

    # 解析
    parser_tool = DocumentParserTool()
    all_docs, errors = parser_tool.parse_files(file_paths)

    if errors:
        for e in errors:
            print(f"  ⚠️  {e}")

    print(f"  解析完成: {len(all_docs)} 个文档（{len(file_paths)} 文件）")

    # 分块
    chunker_cls = get_chunker()
    chunker = chunker_cls()
    chunks = chunker.chunk_all(all_docs)
    print(f"  分块完成: {len(chunks)} 个块")

    # 构建索引
    config = DEFAULT_CONFIG
    embedder = EmbeddingManager(config.embedding_model)
    embeddings = embedder.get_embeddings()

    indexer = IndexManager(embeddings)
    indexer.build_index(chunks)

    retriever = HybridRetriever(indexer, chunks, rrf_k=config.rrf_k)

    # 构建 chunk 索引表（用于判断相关性）
    chunk_index = []
    for i, c in enumerate(chunks):
        source = c.metadata.get("source", "") or c.metadata.get("file_name", "")
        source_file = Path(source).name if source else "unknown"
        chunk_index.append({
            "chunk_id": c.metadata.get("chunk_id", f"chunk_{i}"),
            "index": i,
            "content": c.page_content[:200],
            "source_file": source_file,
        })

    return retriever, chunk_index, chunks


# ============================================================
# 2. 加载 QA benchmark
# ============================================================

def load_qa_benchmark() -> List[Dict]:
    qa_file = ROOT / "data" / "eval" / "qa_benchmark.json"
    if not qa_file.exists():
        print(f"❌ {qa_file} 不存在")
        sys.exit(1)
    with open(qa_file, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 3. 指标计算
# ============================================================

def is_relevant(chunk_source_file: str, relevant_docs: List[str]) -> bool:
    """判断一个 chunk 是否与 QA pair 相关

    如果 chunk 的来源文件名在 QA 的 relevant_docs 列表中，就算相关。
    """
    if not relevant_docs:
        return False
    # 支持多级目录匹配
    for rd in relevant_docs:
        if rd == chunk_source_file:
            return True
        # 也支持子串匹配（处理路径差异）
        if chunk_source_file.endswith(rd) or rd.endswith(chunk_source_file):
            return True
    return False


def calc_recall_at_k(retrieved_chunks: List[Dict], relevant_set: set, k: int) -> float:
    """Recall@k: 前 k 个结果中相关文档数 / 总相关文档数"""
    if not relevant_set:
        return 0.0
    retrieved_at_k = retrieved_chunks[:k]
    hits = sum(1 for c in retrieved_at_k if c["index"] in relevant_set)
    return hits / len(relevant_set)


def calc_precision_at_k(retrieved_chunks: List[Dict], relevant_set: set, k: int) -> float:
    """Precision@k: 前 k 个结果中相关文档数 / k"""
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved_chunks[:k]
    if not retrieved_at_k:
        return 0.0
    hits = sum(1 for c in retrieved_at_k if c["index"] in relevant_set)
    return hits / len(retrieved_at_k)


def calc_mrr(retrieved_chunks: List[Dict], relevant_set: set) -> float:
    """MRR: 第一个相关结果的倒数排名

    如果没找到相关结果，返回 0。
    """
    for rank, c in enumerate(retrieved_chunks, start=1):
        if c["index"] in relevant_set:
            return 1.0 / rank
    return 0.0


def calc_ndcg(retrieved_chunks: List[Dict], relevant_set: set, k: int) -> float:
    """nDCG@k: 归一化折损累计增益"""
    dcg = 0.0
    idcg = 0.0
    for i in range(min(k, len(retrieved_chunks))):
        rel = 1.0 if retrieved_chunks[i]["index"] in relevant_set else 0.0
        dcg += rel / (i + 1)  # log2(1) ≈ 0, log2(2)=1, so i+1 works
    # 理想的排序：所有相关文档排前面
    num_rel = len(relevant_set)
    for i in range(min(k, num_rel)):
        idcg += 1.0 / (i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 4. 主评估流程
# ============================================================

def run_evaluation():
    print("=" * 60)
    print("  检索层评估 — Recall@k / Precision@k / MRR")
    print("=" * 60)
    print()

    # ---- 构建索引 ----
    print("📦 构建 benchmark 索引...")
    t0 = time.time()
    retriever, chunk_index, chunks = build_benchmark_index()
    build_time = time.time() - t0
    print(f"  索引就绪（{build_time:.1f}s）")
    print()

    # ---- 加载 QA ----
    qa_data = load_qa_benchmark()
    print(f"📊 加载了 {len(qa_data)} 个 QA pair")
    print()

    # ---- 计算所有 chunk 跟源文件的关系 ----
    # 建立 "source_file → [chunk_index]" 映射
    file_to_chunks: Dict[str, set] = {}
    for ci in chunk_index:
        fn = ci["source_file"]
        if fn not in file_to_chunks:
            file_to_chunks[fn] = set()
        file_to_chunks[fn].add(ci["index"])

    # ---- 逐条评估 ----
    results = []
    all_recall_3 = []
    all_precision_3 = []
    all_mrr = []

    for qa in qa_data:
        qid = qa["id"]
        question = qa["question"]
        relevant_docs = qa.get("relevant_docs", [])
        difficulty = qa.get("difficulty", "unknown")

        # 找到该 QA 相关的所有 chunk 索引
        relevant_set = set()
        for rd in relevant_docs:
            # 匹配 chunk 来源文件
            for fn, chunk_set in file_to_chunks.items():
                if fn == rd or fn.endswith(rd) or rd.endswith(fn):
                    relevant_set.update(chunk_set)

        if not relevant_set:
            print(f"  ⚠️  {qid}: 未找到相关 chunk（relevant_docs={relevant_docs}），跳过")
            continue

        # 执行混合检索
        retrieved = retriever.hybrid_search(question, top_k=10)

        # 将检索结果映射到 chunk_index
        retrieved_mapped = []
        seen = set()
        for doc in retrieved:
            content = doc.page_content[:200]
            for ci in chunk_index:
                if ci["index"] not in seen and ci["content"] == content:
                    retrieved_mapped.append(ci)
                    seen.add(ci["index"])
                    break

        # 计算指标
        recall_1 = calc_recall_at_k(retrieved_mapped, relevant_set, 1)
        recall_3 = calc_recall_at_k(retrieved_mapped, relevant_set, 3)
        recall_5 = calc_recall_at_k(retrieved_mapped, relevant_set, 5)
        precision_1 = calc_precision_at_k(retrieved_mapped, relevant_set, 1)
        precision_3 = calc_precision_at_k(retrieved_mapped, relevant_set, 3)
        precision_5 = calc_precision_at_k(retrieved_mapped, relevant_set, 5)
        mrr = calc_mrr(retrieved_mapped, relevant_set)
        ndcg_3 = calc_ndcg(retrieved_mapped, relevant_set, 3)

        results.append({
            "id": qid,
            "question": question,
            "difficulty": difficulty,
            "relevant_docs": relevant_docs,
            "total_relevant_chunks": len(relevant_set),
            "recall@1": round(recall_1, 4),
            "recall@3": round(recall_3, 4),
            "recall@5": round(recall_5, 4),
            "precision@1": round(precision_1, 4),
            "precision@3": round(precision_3, 4),
            "precision@5": round(precision_5, 4),
            "mrr": round(mrr, 4),
            "ndcg@3": round(ndcg_3, 4),
        })

        all_recall_3.append(recall_3)
        all_precision_3.append(precision_3)
        all_mrr.append(mrr)

    # ---- 输出报告 ----
    n = len(results)
    if n == 0:
        print("\n❌ 没有可评估的条目")
        return

    avg_recall_3 = sum(all_recall_3) / n
    avg_precision_3 = sum(all_precision_3) / n
    avg_mrr = sum(all_mrr) / n

    print()
    print("=" * 60)
    print("  📊 检索层评估报告")
    print("=" * 60)
    print(f"  评估条目: {n} / {len(qa_data)}")
    print(f"  检索方式: 向量(FAISS) + BM25 + RRF 融合")
    print()
    print(f"  {'指标':>20}  {'分数':>8}")
    print(f"  {'-'*20}  {'-'*8}")
    print(f"  {'Recall@3':>20}  {avg_recall_3:.4f}")
    print(f"  {'Precision@3':>20}  {avg_precision_3:.4f}")
    print(f"  {'MRR':>20}  {avg_mrr:.4f}")
    print()

    # 按难度分层
    print(f"  按难度分层（Recall@3）:")
    print(f"  {'难度':>15}  {'条目':>4}  {'Recall@3':>10}  {'Precision@3':>12}  {'MRR':>8}")
    print(f"  {'-'*15}  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*8}")
    for diff in ["L1_factual", "L2_comparison", "L3_reasoning", "L4_cross_doc"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        diff_label = {"L1_factual": "L1 事实性", "L2_comparison": "L2 比较性",
                      "L3_reasoning": "L3 推理", "L4_cross_doc": "L4 跨文档"}.get(diff, diff)
        r3 = sum(r["recall@3"] for r in subset) / len(subset)
        p3 = sum(r["precision@3"] for r in subset) / len(subset)
        m = sum(r["mrr"] for r in subset) / len(subset)
        print(f"  {diff_label:>15}  {len(subset):>4}  {r3:.4f}{'':>6}  {p3:.4f}{'':>8}  {m:.4f}")

    print()
    print(f"  各条目详情:")
    print(f"  {'ID':>8}  {'Difficulty':>15}  {'R@3':>6}  {'P@3':>6}  {'MRR':>6}  {'Question':<30}")
    print(f"  {'-'*8}  {'-'*15}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*30}")
    for r in results:
        print(f"  {r['id']:>8}  {r['difficulty']:>15}  {r['recall@3']:.4f}  {r['precision@3']:.4f}  {r['mrr']:.4f}  {r['question'][:28]}")

    # ---- 保存结果 ----
    save_path = ROOT / "data" / "eval" / "retrieval_report.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "retrieval": "hybrid (FAISS + BM25 + RRF)",
            "top_k_evaluated": 10,
            "metrics": ["Recall@k", "Precision@k", "MRR", "nDCG@k"],
        },
        "summary": {
            "total_qa": len(qa_data),
            "evaluated": n,
            "avg_recall@3": round(avg_recall_3, 4),
            "avg_precision@3": round(avg_precision_3, 4),
            "avg_mrr": round(avg_mrr, 4),
        },
        "difficulty_breakdown": {},
        "results": results,
    }
    for diff in ["L1_factual", "L2_comparison", "L3_reasoning", "L4_cross_doc"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            report["difficulty_breakdown"][diff] = {
                "count": len(subset),
                "avg_recall@3": round(sum(r["recall@3"] for r in subset) / len(subset), 4),
                "avg_mrr": round(sum(r["mrr"] for r in subset) / len(subset), 4),
            }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"  💾 报告已保存: {save_path}")
    print("=" * 60)

    # ---- 建议 ----
    print()
    print("  💡 结果解读:")
    print(f"  - Recall@3 = {avg_recall_3:.2%}：{n} 条中前 3 个结果平均召回 {avg_recall_3:.0%} 的相关文档")
    print(f"  - MRR = {avg_mrr:.4f}：第一个相关结果平均排在第 {1/avg_mrr:.1f} 位" if avg_mrr > 0 else "  - MRR = 0：没有检索到任何相关结果")
    print()
    if avg_recall_3 < 0.7:
        print("  ⚠️  Recall@3 < 0.7，建议优先优化检索层：")
        print("     - 检查 embedding 模型是否与文档领域匹配")
        print("     - 尝试增大 top_k 或调整 chunk 大小")
        print("     - 考虑加 query 改写/HyDE")
    if avg_mrr < 0.6:
        print("  ⚠️  MRR < 0.6，第一个相关结果排名偏低：")
        print("     - 考虑加 cross-encoder 二次 rerank")
        print("     - 检查 RRF 融合参数（rrf_k=60 是否合适）")


if __name__ == "__main__":
    run_evaluation()
