"""
RAGAS 评估 — Faithfulness / Answer Relevancy / Context Precision / Context Recall

用法：
    # 需要后端在运行 + API Key 已配置
    python tests/eval_ragas.py

输出：
    - data/eval/ragas_report.json（结构化结果）
"""
import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 加载 .env
_env_file = ROOT / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("DASHSCOPE_API_KEY="):
                os.environ["DASHSCOPE_API_KEY"] = _line.split("=", 1)[1].strip().strip('"').strip("'")

import requests

API_URL = "http://localhost:8000"


def call_api(question: str) -> Dict:
    try:
        resp = requests.post(
            f"{API_URL}/api/ask/trace",
            json={"query": question, "file_paths": [], "max_iterations": 2},
            timeout=300,
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("  RAGAS 评估 — Faithfulness / Relevancy / Precision / Recall")
    print("=" * 60)
    print()

    from tests.eval_retrieval import load_qa_benchmark
    all_qa = load_qa_benchmark()
    qa_list = all_qa[:30]
    print(f"\U0001f4cb 加载 {len(all_qa)} 条，评估子集 {len(qa_list)} 条")
    print()

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, qa in enumerate(qa_list):
        qid = qa["id"]
        question = qa["question"]
        gt = qa.get("ground_truth", "")

        sys.stdout.write(f"\r  [{i+1}/{len(qa_list)}] {qid}...")
        sys.stdout.flush()

        api_result = call_api(question)
        if not api_result.get("success"):
            print(f"\n  {qid} API 失败: {api_result.get('error','')}")
            continue

        answer = api_result.get("answer", "")
        chunks = [c.get("content_snippet", "") for c in api_result.get("retrieved_chunks", [])]

        if answer and chunks and gt:
            questions.append(question)
            answers.append(answer)
            contexts.append(chunks)
            ground_truths.append(gt)

    print(f"\n\nOK 成功获取 {len(questions)} 条有效问答")
    print()

    if len(questions) < 3:
        print("有效数据不足 3 条，跳过 RAGAS 评分")
        return

    print("正在计算 RAGAS 指标...")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    try:
        result = evaluate(dataset, metrics=[
            faithfulness, answer_relevancy, context_precision, context_recall,
        ])

        print(f"\n  OK RAGAS 评分完成")
        print(f"\n  {'metric':>25}  {'score':>8}")
        print(f"  {'-'*25}  {'-'*8}")
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = 0
            if hasattr(result, metric):
                val = getattr(result, metric)
            elif isinstance(result, dict):
                val = result.get(metric, 0)
            print(f"  {metric:>25}  {val:.4f}")
        print()

        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": len(questions),
            "metrics": {},
        }
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if hasattr(result, metric):
                report["metrics"][metric] = float(getattr(result, metric))

        save_path = ROOT / "data" / "eval" / "ragas_report.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  Report saved: {save_path}")

    except Exception as e:
        print(f"\n  RAGAS 评分失败: {e}")
        import traceback; traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    main()
