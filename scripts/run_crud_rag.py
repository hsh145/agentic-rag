"""
CRUD-RAG 中文 RAG Benchmark 集成

用法：
    # 1. 下载数据集（一次性）
    python scripts/run_crud_rag.py --download

    # 2. 运行评估（需服务运行中）
    python scripts/run_crud_rag.py --eval --samples 50

    # 3. 全流程
    python scripts/run_crud_rag.py --all --samples 20

数据集说明：
  CRUD-RAG (IAAR-Shanghai) 是一个全面的中文 RAG 评估基准，
  包含 36,166 个测试样本，覆盖 Create/Read/Update/Delete 四种任务。

  来源：https://github.com/IAAR-Shanghai/CRUD_RAG
  论文：https://arxiv.org/abs/2401.17043
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import urlretrieve
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# 配置
# ============================================================
CRUD_RAG_URL = "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/crud_split/split_merged.json"
CRUD_DOCS_URL = "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/80000_docs"

DATA_DIR = Path("data/eval/crud_rag")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_DATASET_PATH = DATA_DIR / "split_merged.json"
LOCAL_RESULTS_PATH = DATA_DIR / "crud_rag_results.json"
LOCAL_REPORT_PATH = DATA_DIR / "crud_rag_report.json"

API_URL = "http://localhost:8000/api/ask"


# ============================================================
# 下载
# ============================================================

def download_dataset():
    """下载 CRUD-RAG 数据集"""
    if LOCAL_DATASET_PATH.exists():
        size = LOCAL_DATASET_PATH.stat().st_size
        print(f"✅ 数据集已存在: {LOCAL_DATASET_PATH} ({size//1024}KB)")
        return True

    print(f"📥 下载 CRUD-RAG 数据集 (~20MB)...")
    print(f"   来源: {CRUD_RAG_URL}")

    try:
        urlretrieve(CRUD_RAG_URL, str(LOCAL_DATASET_PATH))
        size = LOCAL_DATASET_PATH.stat().st_size
        print(f"✅ 下载完成: {LOCAL_DATASET_PATH} ({size//1024}KB)")
        return True
    except HTTPError as e:
        print(f"❌ 下载失败 (HTTP {e.code}): {e.reason}")
        print(f"   请手动下载后放到: {LOCAL_DATASET_PATH}")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


# ============================================================
# 数据加载与适配
# ============================================================

def load_crud_rag_data() -> List[Dict]:
    """加载 CRUD-RAG 数据集，适配为统一评估格式

    CRUD-RAG 数据结构：
    {
      "event_summary": [{"event": "...", "summary": "..."}],     # Delete任务
      "continuing_writing": [{"ID": "...", "event": "...", ...}], # Create任务
      "hallu_modified": [{"ID": "...", "headLine": "...", ...}],  # Update任务
      "questanswer_1doc": [{...}],                                  # Read 1doc
      "questanswer_2docs": [{...}],                                  # Read 2docs
      "questanswer_3docs": [{...}],                                  # Read 3docs
    }
    """
    if not LOCAL_DATASET_PATH.exists():
        print("数据集不存在，请先运行 --download")
        return []

    with open(LOCAL_DATASET_PATH, encoding="utf-8") as f:
        raw_data = json.load(f)

    adapted = []
    type_index = {
        "questanswer_1doc": "read_single",
        "questanswer_2docs": "read_multi",
        "questanswer_3docs": "read_multi",
        "event_summary": "delete",
        "continuing_writing": "create",
        "hallu_modified": "update",
    }

    for task_type, task_data in raw_data.items():
        mapped_type = type_index.get(task_type, task_type)

        if not isinstance(task_data, list):
            continue

        for i, item in enumerate(task_data):
            if not isinstance(item, dict):
                continue

            # 不同任务类型的字段名不同，统一提取
            if task_type.startswith("questanswer"):
                # Read 任务: questions/answers 是单条字符串，不是列表
                question = item.get("questions", "")
                answer = item.get("answers", "")
                if question and answer:
                    adapted.append({
                        "id": f"{item.get('ID', task_type)}",
                        "question": question,
                        "ground_truth": answer,
                        "type": "read",
                        "difficulty": {
                            "questanswer_1doc": "L1_factual",
                            "questanswer_2docs": "L3_reasoning",
                            "questanswer_3docs": "L4_cross_doc",
                        }.get(task_type, "L1_factual"),
                    })
            else:
                # Create/Update/Delete 任务
                question = item.get("event") or item.get("headLine") or ""
                answer = item.get("summary") or item.get("modified_text") or ""

                if question and answer:
                    adapted.append({
                        "id": item.get("ID", f"{task_type}_{i:05d}"),
                        "question": question,
                        "ground_truth": answer,
                        "type": mapped_type,
                        "difficulty": "L4_cross_doc",
                    })

    return adapted


def print_dataset_stats(items: List[Dict]):
    """输出数据集统计"""
    if not items:
        return

    types = {}
    for item in items:
        t = item.get("type", "unknown")
        types[t] = types.get(t, 0) + 1

    print()
    print("CRUD-RAG 数据集统计:")
    print(f"  总计: {len(items)} 条")
    for t, count in sorted(types.items()):
        pct = count / len(items) * 100
        type_name = {
            "read_single": "单文档阅读",
            "read_multi": "多文档阅读",
            "create": "续写",
            "update": "纠错",
            "delete": "多文档摘要",
        }.get(t, t)
        print(f"  {type_name:>12}: {count:>6} ({pct:5.1f}%)")
    print()


# ============================================================
# 评估
# ============================================================

def evaluate_with_api(item: Dict, api_url: str = API_URL, timeout: int = 120) -> Dict:
    """使用运行中的 RAG 服务评估一个问题"""
    import requests

    t0 = time.time()
    try:
        resp = requests.post(
            api_url,
            json={
                "query": item["question"],
                "max_iterations": 2,
            },
            timeout=timeout,
        )
        elapsed = time.time() - t0
        result = resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "id": item["id"],
            "success": False,
            "error": f"无法连接 {api_url}，请确认服务已启动",
            "elapsed": time.time() - t0,
        }
    except requests.exceptions.Timeout:
        return {
            "id": item["id"],
            "success": False,
            "error": "请求超时",
            "elapsed": time.time() - t0,
        }
    except Exception as e:
        return {
            "id": item["id"],
            "success": False,
            "error": str(e),
            "elapsed": time.time() - t0,
        }

    if not result.get("success"):
        return {
            "id": item["id"],
            "success": False,
            "error": result.get("error", "未知错误"),
            "elapsed": elapsed,
        }

    return {
        "id": item["id"],
        "success": True,
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "answer": result.get("answer", ""),
        "elapsed": elapsed,
        "type": item.get("type", "read"),
    }


def run_evaluation(sample_size: Optional[int] = None):
    """运行评估"""
    if not LOCAL_DATASET_PATH.exists():
        print(f"❌ 数据集不存在，请先运行 --download")
        return

    items = load_crud_rag_data()
    if not items:
        return

    print_dataset_stats(items)

    # 采样
    if sample_size and sample_size < len(items):
        import random
        random.seed(42)
        items = random.sample(items, sample_size)
        print(f"🎲 随机采样 {sample_size} 条")
    else:
        print(f"📋 全量评估 {len(items)} 条（这可能需要很长时间）")

    print()
    print(f"🔗 API: {API_URL}")
    print(f"{'='*50}")
    print()

    # 逐条评估
    results = []
    success_count = 0
    fail_count = 0
    total_time = 0

    for i, item in enumerate(items):
        print(f"\r  [{i+1}/{len(items)}] {item['question'][:40]}...", end="", flush=True)

        result = evaluate_with_api(item)
        results.append(result)

        if result["success"]:
            success_count += 1
            total_time += result.get("elapsed", 0)
        else:
            fail_count += 1
            print(f"\n  ❌ [{result['id']}] {result.get('error', '')}")

    print(f"\n\n{'='*50}")
    print(f"  CRUD-RAG 评估完成")
    print(f"{'='*50}")
    print(f"  总计: {len(results)} 条")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    if success_count:
        print(f"  平均耗时: {total_time/success_count:.1f}s")
        print(f"  成功率: {success_count/len(results)*100:.1f}%")

    # 保存
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "success": success_count,
        "failed": fail_count,
        "avg_time": round(total_time / max(success_count, 1), 2),
        "success_rate": round(success_count / max(len(results), 1) * 100, 1),
    }

    with open(LOCAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(LOCAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  详情已保存:")
    print(f"    结果: {LOCAL_RESULTS_PATH}")
    print(f"    报告: {LOCAL_REPORT_PATH}")
    print()

    return report


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CRUD-RAG 中文 RAG Benchmark 评估")
    parser.add_argument("--download", action="store_true", help="下载数据集")
    parser.add_argument("--eval", action="store_true", help="运行评估")
    parser.add_argument("--all", action="store_true", help="下载+评估全流程")
    parser.add_argument("--samples", type=int, default=None,
                        help="采样数量（默认全量）")
    parser.add_argument("--stats", action="store_true", help="仅显示数据集统计")

    args = parser.parse_args()

    # 默认行为：显示帮助
    if not any([args.download, args.eval, args.all, args.stats]):
        parser.print_help()
        return

    # 下载
    if args.download or args.all:
        if not download_dataset():
            return

    # 统计
    if args.stats or args.all:
        items = load_crud_rag_data()
        print_dataset_stats(items)

    # 评估
    if args.eval or args.all:
        run_evaluation(args.samples)


if __name__ == "__main__":
    main()
