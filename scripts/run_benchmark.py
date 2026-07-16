"""
Agentic RAG 一键基准测试

用法：
    # 端到端延迟
    python scripts/run_benchmark.py

    # 全量测试（解析器 + RAG + 性能）
    python scripts/run_benchmark.py --all

    # 仅解析器测试
    python scripts/run_benchmark.py --parser-only

    # 仅 RAG 检索测试
    python scripts/run_benchmark.py --rag-only

    # 仅端到端性能
    python scripts/run_benchmark.py --perf-only

输出：
    汇总 Markdown 表格，保存在 data/eval/benchmark_report.md
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from statistics import mean, median, stdev

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_CONFIG


# ============================================================
# 端到端性能测试
# ============================================================

async def bench_end_to_end(queries: List[str], iterations: int = 3) -> Dict:
    """端到端延迟测试"""
    from agent.graph import run_agent

    latencies = []
    token_counts = []
    chunk_counts = []

    for q in queries:
        for i in range(iterations):
            t0 = time.time()
            result = await run_agent(query=q, config=DEFAULT_CONFIG)
            elapsed = time.time() - t0

            latencies.append(elapsed)
            final_answer = result.get("final_answer", "")
            chunk_counts.append(len(result.get("retrieved_chunks", [])))
            token_counts.append(len(final_answer))

    latencies.sort()
    n = len(latencies)
    return {
        "samples": n,
        "p50": median(latencies),
        "p95": latencies[int(n * 0.95)] if n > 1 else latencies[-1],
        "p99": latencies[int(n * 0.99)] if n > 1 else latencies[-1],
        "mean": mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "std": stdev(latencies) if n > 1 else 0,
        "avg_chunks": mean(chunk_counts) if chunk_counts else 0,
        "avg_tokens": mean(token_counts) if token_counts else 0,
    }


# ============================================================
# 解析器性能测试
# ============================================================

def bench_parser():
    """解析器性能基准"""
    from parser.text_parser import TextParser
    from parser.detector import FileTypeDetector

    benchmark_dir = Path("data/benchmark")
    if not benchmark_dir.exists():
        return {"error": "benchmark 目录不存在", "results": []}

    parser = TextParser()
    results = []

    for file_path in sorted(benchmark_dir.iterdir()):
        if not file_path.is_file():
            continue
        ftype = FileTypeDetector.detect(str(file_path))
        if ftype == "unknown":
            continue

        t0 = time.time()
        try:
            docs = parser.parse(str(file_path))
            elapsed = time.time() - t0
            content_len = len(docs[0].page_content) if docs else 0
            results.append({
                "file": file_path.name,
                "type": ftype,
                "time": round(elapsed, 3),
                "docs": len(docs),
                "chars": content_len,
                "success": True,
            })
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "file": file_path.name,
                "type": ftype,
                "time": round(elapsed, 3),
                "docs": 0,
                "chars": 0,
                "success": False,
                "error": str(e),
            })

    return {"results": results}


# ============================================================
# RAG 检索性能测试
# ============================================================

def bench_rag_retrieval():
    """RAG 检索性能拆解测试"""
    from rag.embedder import EmbeddingManager
    from rag.indexer import IndexManager
    from langchain_core.documents import Document

    # 构造测试数据
    test_docs = [
        Document(page_content=f"测试文档 {i} 的内容。" * 20, metadata={"source": f"doc{i}.txt"})
        for i in range(10)
    ]

    embedder = EmbeddingManager(DEFAULT_CONFIG.embedding_model)
    indexer = IndexManager(embedder.get_embeddings())

    # 构建索引
    t0 = time.time()
    indexer.build_index(test_docs)
    index_time = time.time() - t0

    # 检索测试
    queries = ["测试", "文档", "检索", "嵌入", "模型"]
    search_times = []

    for q in queries:
        t0 = time.time()
        indexer.similarity_search(q, k=3)
        search_times.append(time.time() - t0)

    return {
        "index_time": round(index_time, 3),
        "search_p50": round(median(search_times), 4),
        "search_mean": round(mean(search_times), 4),
        "num_docs": len(test_docs),
        "embedding_dim": indexer.index.d if indexer.index else 0,
    }


# ============================================================
# 报告生成
# ============================================================

def print_parser_report(data: Dict):
    """输出解析器基准报告"""
    if "error" in data:
        print(f"  ❌ {data['error']}")
        return

    results = data["results"]
    if not results:
        print("  ⚠️ 无测试文件")
        return

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n  📄 解析器性能基准")
    print(f"  {'='*50}")
    print(f"  文件数: {len(results)} | 成功: {len(success)} | 失败: {len(failed)}")
    if success:
        avg_time = mean(r["time"] for r in success)
        print(f"  平均解析耗时: {avg_time:.3f}s")
        print()
        for r in success:
            print(f"    {r['file']:>20} | {r['type']:>8} | "
                  f"{r['time']:.3f}s | {r['chars']:>5} 字符 | "
                  f"{r['docs']} 文档")
    if failed:
        print()
        for r in failed:
            print(f"    ❌ {r['file']}: {r.get('error', 'unknown')}")


def print_perf_report(data: Dict):
    """输出端到端性能报告"""
    print(f"\n  ⚡ 端到端性能基准")
    print(f"  {'='*50}")
    print(f"  采样数: {data['samples']}")
    print(f"  P50:    {data['p50']:.2f}s")
    print(f"  P95:    {data['p95']:.2f}s")
    print(f"  P99:    {data['p99']:.2f}s")
    print(f"  均值:   {data['mean']:.2f}s")
    print(f"  标准差: {data['std']:.2f}s")
    print(f"  最慢:   {data['max']:.2f}s")
    print(f"  最快:   {data['min']:.2f}s")
    print(f"  平均检索块数: {data['avg_chunks']:.1f}")
    print(f"  平均回答长度: {data['avg_tokens']:.0f} 字符")


def print_rag_report(data: Dict):
    """输出 RAG 检索性能报告"""
    print(f"\n  🔍 RAG 检索性能")
    print(f"  {'='*50}")
    print(f"  索引构建: {data['index_time']}s ({data['num_docs']} 文档)")
    print(f"  检索 P50: {data['search_p50']}s")
    print(f"  检索均值: {data['search_mean']}s")
    print(f"  Embedding 维度: {data['embedding_dim']}")


def save_markdown_report(perf_data, parser_data, rag_data):
    """保存 Markdown 格式报告"""
    report = [
        "# Agentic RAG 基准测试报告",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 端到端性能",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| P50 | {perf_data.get('p50', 'N/A')}s |",
        f"| P95 | {perf_data.get('p95', 'N/A')}s |",
        f"| P99 | {perf_data.get('p99', 'N/A')}s |",
        f"| 均值 | {perf_data.get('mean', 'N/A')}s |",
        f"| 采样数 | {perf_data.get('samples', 'N/A')} |",
        "",
    ]

    if "results" in parser_data:
        success = [r for r in parser_data["results"] if r.get("success")]
        failed = [r for r in parser_data["results"] if not r.get("success")]
        report += [
            "## 解析器基准",
            f"| 指标 | 值 |",
            f"|------|----|",
            f"| 总文件数 | {len(parser_data['results'])} |",
            f"| 成功率 | {len(success)}/{len(parser_data['results'])} |",
        ]
        if success:
            report += [
                "",
                "### 各文件解析",
                "| 文件 | 类型 | 耗时 | 字符数 |",
                "|------|------|------|--------|",
            ]
            for r in success:
                report.append(f"| {r['file']} | {r['type']} | {r['time']}s | {r['chars']} |")

    report.append("")
    report.append("---")
    report.append("*由 scripts/run_benchmark.py 自动生成*")

    report_path = Path("data/eval/benchmark_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n  📝 报告已保存: {report_path}")


# ============================================================
# 主入口
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Agentic RAG 基准测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--parser-only", action="store_true", help="仅解析器测试")
    parser.add_argument("--rag-only", action="store_true", help="仅 RAG 检索测试")
    parser.add_argument("--perf-only", action="store_true", help="仅端到端性能测试")
    args = parser.parse_args()

    # 默认：只跑性能
    run_all = args.all or not any([args.parser_only, args.rag_only, args.perf_only])

    print()
    print("=" * 55)
    print("  Agentic RAG 基准测试")
    print("=" * 55)

    perf_data = {}
    parser_data = {}
    rag_data = {}

    # 1. 解析器
    if args.parser_only or run_all:
        print("\n[1/3] 解析器基准...")
        parser_data = bench_parser()
        print_parser_report(parser_data)

    # 2. RAG 检索
    if args.rag_only or run_all:
        print("\n[2/3] RAG 检索基准...")
        rag_data = bench_rag_retrieval()
        print_rag_report(rag_data)

    # 3. 端到端
    if args.perf_only or run_all:
        print("\n[3/3] 端到端性能基准...")
        queries = [
            "什么是SFT微调？",
            "LoRA和Full Fine-tuning的区别是什么？",
        ]
        perf_data = await bench_end_to_end(queries, iterations=2)
        print_perf_report(perf_data)

    print()
    print(f"  {'='*55}")
    print()

    # 保存报告
    if run_all:
        save_markdown_report(perf_data, parser_data, rag_data)


if __name__ == "__main__":
    asyncio.run(main())
