"""
RAGAS 评估脚本 — 计算 Faithfulness、Answer Relevancy、Context Recall、Context Precision

用法：
    pip install ragas datasets
    python tests/eval_ragas.py

输出：
    4 个核心指标的分数 + 按难度的分层分析
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_qa_benchmark(path: str = "data/eval/qa_benchmark.json") -> List[dict]:
    """加载 QA benchmark 数据集"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_system(qa: dict, config_name: str = "default") -> dict:
    """用当前 RAG 系统回答一个问题

    Args:
        qa: QA pair from benchmark
        config_name: 配置名称（用于多个配置对比）

    Returns:
        {"answer": str, "contexts": [str], "config": str}
    """
    # 延迟导入，避免启动时加载所有依赖
    import asyncio
    from agent.graph import run_agent
    from config import DEFAULT_CONFIG

    # 异步运行 agent
    async def _ask():
        result = await run_agent(
            query=qa["question"],
            config=DEFAULT_CONFIG,
        )
        chunks = result.get("retrieved_chunks", [])
        contexts = [
            c.get("content", "") for c in chunks
        ] if chunks else []
        return {
            "answer": result.get("final_answer", ""),
            "contexts": contexts,
            "config": config_name,
        }

    return asyncio.run(_ask())


def run_evaluation():
    """运行 RAGAS 评估"""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"❌ 请安装依赖: pip install ragas datasets")
        print(f"   错误: {e}")
        return

    qa_data = load_qa_benchmark()
    if not qa_data:
        print("❌ 未找到 QA benchmark 数据")
        return

    print(f"📊 加载了 {len(qa_data)} 个 QA pair")
    print(f"   {'='*50}")
    print()

    # 逐条运行系统
    questions, answers, contexts, ground_truths = [], [], [], []
    errors = []

    for i, qa in enumerate(qa_data):
        try:
            sys.stdout.write(f"\r  🔄 [{i+1}/{len(qa_data)}] {qa['question'][:40]}...  ")
            sys.stdout.flush()

            result = run_system(qa)
            questions.append(qa["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(qa["ground_truth"])
        except Exception as e:
            print(f"\n  ❌ {qa['question'][:30]}: {e}")
            errors.append(qa["id"])

    print(f"\n  ✅ 完成 {len(questions) - len(errors)}/{len(qa_data)} 条")
    print()

    if not answers:
        print("❌ 无可用结果")
        return

    # 构建 HuggingFace Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # 运行 RAGAS 评估
    print("📏 正在计算 RAGAS 指标...")
    print("   (首次运行会下载 judge 模型或调用 API)")
    print()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        # 输出结果
        df = result.to_pandas()
        print()
        print("=" * 55)
        print("  RAGAS 评估结果")
        print("=" * 55)
        for col in df.columns:
            if col in ("question", "answer", "contexts", "ground_truth"):
                continue
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  {col:>25}: {mean_val:.4f} (±{std_val:.4f})")
        print("=" * 55)
        print()

        # 按难度分层分析
        difficulty_map = {d["id"]: d["difficulty"] for d in qa_data}
        print("  按难度分层:")
        for diff_level in ["L1_factual", "L2_comparison", "L3_reasoning", "L4_cross_doc"]:
            indices = [i for i, q in enumerate(qa_data) if q["id"] in difficulty_map and difficulty_map.get(q["id"]) == diff_level]
            if not indices:
                continue
            subset = dataset.select(indices)
            diff_label = {
                "L1_factual": "L1 事实性",
                "L2_comparison": "L2 比较性",
                "L3_reasoning": "L3 推理",
                "L4_cross_doc": "L4 跨文档",
            }.get(diff_level, diff_level)

            try:
                sub_result = evaluate(dataset=subset, metrics=[faithfulness, context_recall])
                sub_df = sub_result.to_pandas()
                fth = sub_df["faithfulness"].mean() if "faithfulness" in sub_df.columns else 0
                rec = sub_df["context_recall"].mean() if "context_recall" in sub_df.columns else 0
                print(f"    {diff_label:>15}: faithfulness={fth:.3f}, recall={rec:.3f} ({len(indices)}条)")
            except Exception:
                pass

        print()
        print(f"  结果已保存到: data/eval/ragas_result.csv")
        df.to_csv("data/eval/ragas_result.csv", index=False, encoding="utf-8")

    except Exception as e:
        print(f"❌ RAGAS 评估失败: {e}")
        print()
        print("💡 提示: RAGAS 需要访问 LLM 作为 judge。你可以:")
        print("   1. 设置 OPENAI_API_KEY 环境变量")
        print("   2. 或修改 ragas 配置使用本地模型")
        print("   3. 也可以先用 eval_ragas_local.py 用 qwen-turbo 做 judge")


if __name__ == "__main__":
    run_evaluation()
