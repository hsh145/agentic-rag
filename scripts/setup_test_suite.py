"""
综合测试套件搭建脚本

一键完成：
  1. 生成文档解析测试文件 (PDF/Word/Excel/图片)
  2. 提取 CRUD-RAG 知识库文档到 data/docs/
  3. 生成 RAG QA 评估数据集
  4. 统计展示测试集概览

用法:
    python scripts/setup_test_suite.py          # 全流程
    python scripts/setup_test_suite.py --parser-only   # 仅生成解析测试文件
    python scripts/setup_test_suite.py --rag-only      # 仅搭建 RAG 测试集
    python scripts/setup_test_suite.py --stats         # 仅显示统计
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BENCHMARK_DIR = Path("data/benchmark")
DOCS_DIR = Path("data/docs")
EVAL_DIR = Path("data/eval")
CRUD_DATA_PATH = EVAL_DIR / "crud_rag" / "split_merged.json"

os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ================================================================
# 第1部分：生成文档解析测试文件
# ================================================================

def generate_parser_test_files():
    """生成用于测试解析器的 PDF/Word/Excel/图片 文件"""
    print("\n" + "="*55)
    print("  第1部分：生成文档解析测试文件")
    print("="*55)

    # 导入项目自带的生成脚本
    from scripts.generate_benchmark_files import (
        generate_pdf, generate_word, generate_excel, generate_image
    )

    generate_pdf()
    generate_word()
    generate_excel()
    generate_image()

    # 列出生成的文件
    print("\n  生成的文件:")
    for f in sorted(BENCHMARK_DIR.iterdir()):
        if f.is_file() and f.suffix in ('.pdf', '.docx', '.xlsx', '.png', '.txt', '.md', '.json'):
            size = f.stat().st_size
            print(f"    {f.name:30s} {size:>7,} bytes")


# ================================================================
# 第2部分：提取 CRUD-RAG 知识库文档
# ================================================================

def extract_crud_knowledge_base():
    """从 CRUD-RAG 的 QA 数据中提取 unique 新闻文档到 data/docs/"""
    print("\n" + "="*55)
    print("  第2部分：提取 CRUD-RAG 知识库文档")
    print("="*55)

    if not CRUD_DATA_PATH.exists():
        print(f"  ❌ CRUD-RAG 数据集不存在: {CRUD_DATA_PATH}")
        print(f"  请先运行: python scripts/run_crud_rag.py --download")
        return 0

    with open(CRUD_DATA_PATH, encoding="utf-8") as f:
        raw_data = json.load(f)

    # 从所有任务类型中提取 unique 文档
    all_docs = {}  # hash -> {"id": ..., "text": ...}
    total_items = 0

    for task_type in raw_data:
        items = raw_data[task_type]
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            total_items += 1
            # questanswer 和 event_summary 都有 news1/news2/news3
            for key in ["news1", "news2", "news3"]:
                text = (item.get(key) or "").strip()
                if len(text) > 50:
                    h = hash(text[:500])
                    if h not in all_docs:
                        all_docs[h] = {
                            "id": f"doc_{len(all_docs):05d}",
                            "text": text,
                        }

    print(f"  总数据条目: {total_items}")
    print(f"  提取 unique 文档: {len(all_docs)}")

    # 保存到 data/docs/ 目录
    docs_written = 0
    for h, doc in all_docs.items():
        file_name = f"{doc['id']}.txt"
        file_path = DOCS_DIR / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        docs_written += 1

    print(f"  文档已写入: {docs_written} 个文件到 {DOCS_DIR}")

    # 保存文档索引
    doc_index = []
    for h, doc in all_docs.items():
        doc_index.append({
            "doc_id": doc["id"],
            "file": f"{doc['id']}.txt",
            "chars": len(doc["text"]),
        })
    index_path = DOCS_DIR / "_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    print(f"  文档索引已保存: {index_path}")

    total_chars = sum(len(doc["text"]) for doc in all_docs.values())
    print(f"  知识库总字符数: {total_chars:,}")
    print(f"  平均每篇: {total_chars // max(len(all_docs), 1):,} 字符")

    return len(all_docs)


# ================================================================
# 第3部分：生成 RAG QA 评估数据集
# ================================================================

def generate_rag_qa_benchmark(sample_size: int = 200):
    """从 CRUD-RAG 的 questanswer 数据生成 QA 评估集

    Args:
        sample_size: 每类采样的最大数量
    """
    print("\n" + "="*55)
    print("  第3部分：生成 RAG QA 评估数据集")
    print("="*55)

    if not CRUD_DATA_PATH.exists():
        print(f"  ❌ CRUD-RAG 数据集不存在")
        return

    with open(CRUD_DATA_PATH, encoding="utf-8") as f:
        raw_data = json.load(f)

    import random
    random.seed(42)

    qa_pairs = []
    for task_type in ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]:
        items = raw_data.get(task_type, [])
        if not isinstance(items, list):
            continue

        if len(items) > sample_size:
            items = random.sample(items, sample_size)

        difficulty_map = {
            "questanswer_1doc": "L1_factual",
            "questanswer_2docs": "L3_reasoning",
            "questanswer_3docs": "L4_cross_doc",
        }
        difficulty = difficulty_map[task_type]

        for item in items:
            question = (item.get("questions") or "").strip()
            answer = (item.get("answers") or "").strip()
            if question and answer and len(question) > 5 and len(answer) > 5:
                # 找出引用到的新闻文档
                relevant_docs = []
                for key in ["news1", "news2", "news3"]:
                    text = (item.get(key) or "").strip()
                    if text:
                        doc_id = f"doc_{hash(text[:500])}"
                        relevant_docs.append(f"{doc_id}.txt")

                qa_pairs.append({
                    "id": f"{task_type}_{item.get('ID', '')}",
                    "question": question,
                    "ground_truth": answer,
                    "relevant_docs": relevant_docs,
                    "difficulty": difficulty,
                    "expected_topics": [],
                })

    # 添加原有的 sample 文件 QA（保持兼容）
    existing_qa = EVAL_DIR / "qa_benchmark.json"
    if existing_qa.exists():
        with open(existing_qa, encoding="utf-8") as f:
            try:
                original = json.load(f)
                if isinstance(original, list):
                    qa_pairs = original + qa_pairs
                    print(f"  合并了原有 {len(original)} 条 QA")
            except json.JSONDecodeError:
                pass

    # 保存
    output_path = EVAL_DIR / "qa_benchmark.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"  总 QA 对数: {len(qa_pairs)}")
    # 按难度统计
    by_difficulty = {}
    for qa in qa_pairs:
        d = qa.get("difficulty", "unknown")
        by_difficulty[d] = by_difficulty.get(d, 0) + 1
    for d, count in sorted(by_difficulty.items()):
        print(f"    {d}: {count} 条")
    print(f"  已保存: {output_path}")


# ================================================================
# 统计展示
# ================================================================

def print_stats():
    """输出测试集整体统计"""
    print("\n" + "="*55)
    print("  Agentic RAG 测试集概览")
    print("="*55)

    # 1. 解析器测试文件
    print("\n  📄 解析器测试文件:")
    if BENCHMARK_DIR.exists():
        total_files = 0
        success_files = 0
        for f in sorted(BENCHMARK_DIR.iterdir()):
            if f.is_file() and f.suffix in ('.pdf', '.docx', '.xlsx', '.png', '.txt', '.md', '.json'):
                size = f.stat().st_size
                total_files += 1
                print(f"    {f.name:30s} {size:>8,} bytes")
                success_files += 1
        print(f"    {'─'*40}")
        print(f"    {'总计':>20s}: {total_files} 个文件")
    else:
        print(f"    ❌ 目录不存在，请先运行生成脚本")

    # 2. 知识库文档
    print("\n  📚 RAG 知识库:")
    if DOCS_DIR.exists():
        doc_files = [f for f in DOCS_DIR.iterdir() if f.suffix == '.txt']
        total_chars = sum(f.stat().st_size for f in doc_files)
        print(f"    文档数: {len(doc_files)}")
        print(f"    总大小: {total_chars:,} 字符")
        if doc_files:
            avg_size = total_chars // len(doc_files)
            print(f"    平均大小: {avg_size:,} 字符/篇")

        # 检查是否有索引
        index_path = DOCS_DIR / "_index.json"
        if index_path.exists():
            print(f"    文档索引: ✅ 存在")
        else:
            print(f"    文档索引: ❌ 未生成")
    else:
        print(f"    ❌ 目录不存在")

    # 3. QA 评估集
    print("\n  ❓ QA 评估集:")
    qa_path = EVAL_DIR / "qa_benchmark.json"
    if qa_path.exists():
        with open(qa_path, encoding="utf-8") as f:
            try:
                qa_data = json.load(f)
                print(f"    QA 对数: {len(qa_data)}")
                by_diff = {}
                for qa in qa_data:
                    d = qa.get("difficulty", "unknown")
                    by_diff[d] = by_diff.get(d, 0) + 1
                for d, c in sorted(by_diff.items()):
                    print(f"      {d}: {c} 条")
                # 平均问题长度
                avg_q = sum(len(qa.get("question", "")) for qa in qa_data) // max(len(qa_data), 1)
                avg_a = sum(len(qa.get("ground_truth", "")) for qa in qa_data) // max(len(qa_data), 1)
                print(f"    平均问题长度: {avg_q} 字符")
                print(f"    平均答案长度: {avg_a} 字符")
            except json.JSONDecodeError:
                print(f"    ❌ JSON 格式错误")
    else:
        print(f"    ❌ 未找到 QA 评估集")

    # 4. FAISS 索引
    print("\n  🔍 FAISS 索引:")
    index_file = Path("data/index/index.faiss")
    docs_file = Path("data/index/documents.pkl")
    if index_file.exists() and docs_file.exists():
        size_mb = index_file.stat().st_size / 1024 / 1024
        import pickle
        with open(docs_file, "rb") as f:
            docs = pickle.load(f)
        print(f"    索引文件: {size_mb:.1f} MB")
        print(f"    文档块数: {len(docs)}")
        total_chars = sum(len(d.page_content) for d in docs)
        print(f"    总字符数: {total_chars:,}")
    else:
        print(f"    ❌ 未构建索引（或仅部分存在）")

    print(f"\n  {'='*55}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="综合测试套件搭建")
    parser.add_argument("--parser-only", action="store_true", help="仅生成解析测试文件")
    parser.add_argument("--rag-only", action="store_true", help="仅搭建 RAG 测试集")
    parser.add_argument("--stats", action="store_true", help="仅显示统计")
    parser.add_argument("--samples", type=int, default=200, help="每类 QA 采样数 (默认200)")
    args = parser.parse_args()

    run_all = not any([args.parser_only, args.rag_only, args.stats])

    t0 = time.time()

    if args.stats:
        print_stats()
        return

    if args.parser_only or run_all:
        generate_parser_test_files()

    if args.rag_only or run_all:
        count = extract_crud_knowledge_base()
        if count > 0:
            generate_rag_qa_benchmark(sample_size=args.samples)

    elapsed = time.time() - t0
    print(f"\n  ✅ 完成! 耗时: {elapsed:.1f}s")

    # 最后显示统计
    print_stats()


if __name__ == "__main__":
    main()
