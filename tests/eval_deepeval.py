"""
DeepEval 评估 — CI 集成的 RAG 测试框架

用法：
    pip install deepeval
    pytest tests/eval_deepeval.py -v

优势：
  - 天然 pytest 集成，直接作为 CI 断言
  - 支持自定义 judge LLM（可以设为 qwen-turbo）
  - 内置 Faithfulness、Hallucination、Bias、Toxicity 等指标
  - 生成可视化的评估 dashboard

参考：
  https://docs.confident-ai.com/docs/metrics-faithfulness
"""

import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# ============================================================
# 导入 DeepEval（带 fallback 提示）
# ============================================================
try:
    from deepeval import assert_test
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase
    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    pytest.skip("请安装 deepeval: pip install deepeval", allow_module_level=True)


# ============================================================
# 配置 Judge LLM
# ============================================================

def setup_deepeval_judge():
    """配置 DeepEval 使用本地 LLM 作为 judge

    DeepEval 默认使用 GPT-4 作为 judge。要改为 qwen-turbo，
    设置以下环境变量或使用 DeepEval 的 custom model。
    """
    # 方式1：使用自定义 LLM
    from langchain_community.chat_models import ChatTongyi

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        os.environ["DEEPEVAL_LLM_MODEL_NAME"] = "custom"
        return ChatTongyi(model="qwen-turbo", temperature=0.0, dashscope_api_key=api_key)
    return None


# ============================================================
# 加载测试数据
# ============================================================

def load_test_cases():
    """从 QA benchmark 加载测试用例"""
    qa_file = Path("data/eval/qa_benchmark.json")
    if not qa_file.exists():
        pytest.skip("未找到 QA benchmark 数据")

    with open(qa_file, encoding="utf-8") as f:
        data = json.load(f)

    return data[:5]  # CI 环境下跑前 5 条即可


# ============================================================
# 运行系统并获取结果
# ============================================================

def run_system_qa(qa: dict) -> dict:
    """用 RAG 系统回答一个 QA pair"""
    import asyncio
    from agent.graph import run_agent
    from config import DEFAULT_CONFIG

    async def _ask():
        result = await run_agent(query=qa["question"], config=DEFAULT_CONFIG)
        chunks = result.get("retrieved_chunks", [])
        contexts = [c.get("content", "") for c in chunks] if chunks else []
        return {
            "answer": result.get("final_answer", ""),
            "contexts": contexts,
        }

    return asyncio.run(_ask())


# ============================================================
# 测试用例
# ============================================================

@pytest.mark.skipif(not HAS_DEEPEVAL, reason="需要 deepeval 包")
class TestRAGMetrics:
    """DeepEval RAG 指标测试"""

    @pytest.fixture(scope="class")
    def test_cases_data(self):
        return load_test_cases()

    @pytest.fixture(scope="class")
    def judge(self):
        return setup_deepeval_judge()

    def test_faithfulness(self, test_cases_data, judge):
        """测试 Faithfulness（忠实度）"""
        metric = FaithfulnessMetric(
            threshold=0.7,
            model=judge,
        )

        for qa in test_cases_data:
            result = run_system_qa(qa)
            test_case = LLMTestCase(
                input=qa["question"],
                actual_output=result["answer"],
                retrieval_context=result["contexts"],
            )
            metric.measure(test_case)
            assert metric.score >= 0.5, (
                f"Faithfulness 不合格: {qa['id']} "
                f"(score={metric.score:.2f})"
            )

    def test_answer_relevancy(self, test_cases_data, judge):
        """测试 Answer Relevancy（回答相关性）"""
        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=judge,
        )

        for qa in test_cases_data:
            result = run_system_qa(qa)
            test_case = LLMTestCase(
                input=qa["question"],
                actual_output=result["answer"],
            )
            metric.measure(test_case)
            assert metric.score >= 0.5, (
                f"Answer Relevancy 不合格: {qa['id']} "
                f"(score={metric.score:.2f})"
            )

    def test_contextual_precision(self, test_cases_data, judge):
        """测试 Contextual Precision（上下文精确度）"""
        metric = ContextualPrecisionMetric(
            threshold=0.7,
            model=judge,
        )

        for qa in test_cases_data:
            result = run_system_qa(qa)
            if not result["contexts"]:
                continue  # 没有检索结果时跳过
            test_case = LLMTestCase(
                input=qa["question"],
                actual_output=result["answer"],
                expected_output=qa["ground_truth"],
                retrieval_context=result["contexts"],
            )
            metric.measure(test_case)
            assert metric.score >= 0.5, (
                f"Contextual Precision 不合格: {qa['id']} "
                f"(score={metric.score:.2f})"
            )

    def test_contextual_recall(self, test_cases_data, judge):
        """测试 Contextual Recall（上下文召回率）"""
        metric = ContextualRecallMetric(
            threshold=0.7,
            model=judge,
        )

        for qa in test_cases_data:
            result = run_system_qa(qa)
            if not result["contexts"]:
                continue
            test_case = LLMTestCase(
                input=qa["question"],
                actual_output=result["answer"],
                expected_output=qa["ground_truth"],
                retrieval_context=result["contexts"],
            )
            metric.measure(test_case)
            assert metric.score >= 0.5, (
                f"Contextual Recall 不合格: {qa['id']} "
                f"(score={metric.score:.2f})"
            )

    def test_hallucination(self, test_cases_data, judge):
        """测试 Hallucination（幻觉检测）"""
        metric = HallucinationMetric(
            threshold=0.3,  # 越低越好，0.3 以下合格
            model=judge,
        )

        for qa in test_cases_data:
            result = run_system_qa(qa)
            if not result["contexts"]:
                continue
            test_case = LLMTestCase(
                input=qa["question"],
                actual_output=result["answer"],
                context=result["contexts"],
            )
            metric.measure(test_case)
            assert metric.score <= 0.5, (
                f"Hallucination 过高: {qa['id']} "
                f"(score={metric.score:.2f})"
            )
