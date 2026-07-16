"""
本地 RAGAS 评估 — 使用 qwen-turbo 作为 judge，不依赖 OpenAI

用法：
    python tests/eval_ragas_local.py

说明：
    标准 RAGAS 默认使用 OpenAI 作为 judge LLM。
    本脚本通过自定义评判方式，使用 DashScope 的 qwen-turbo 进行打分。
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# LLM Judge 封装
# ============================================================

class QwenJudge:
    """使用 qwen-turbo 作为评估 judge"""

    def __init__(self, model: str = "qwen-turbo"):
        from langchain_community.chat_models import ChatTongyi
        from langchain_core.messages import HumanMessage

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            api_key = os.environ.get("MOONSHOT_API_KEY", "")
        self.llm = ChatTongyi(model=model, temperature=0.0, dashscope_api_key=api_key)
        self.HumanMessage = HumanMessage

    def ask(self, prompt: str) -> str:
        resp = self.llm.invoke([self.HumanMessage(content=prompt)])
        return resp.content.strip()


# ============================================================
# 评估函数
# ============================================================

def eval_faithfulness(judge: QwenJudge, question: str, answer: str, contexts: List[str]) -> float:
    """评估 Faithfulness：回答是否忠于检索到的上下文"""
    context_text = "\n\n".join(contexts) if contexts else "（无上下文）"

    prompt = f"""你是评估助手。请判断以下回答是否忠实地基于给定的参考信息，没有编造内容。

【参考信息】
{context_text[:3000]}

【用户问题】
{question}

【模型回答】
{answer}

请输出一个 0 到 1 之间的分数：
- 1.0 = 完全基于参考信息，无编造
- 0.5 = 部分基于参考信息，有少量推断
- 0.0 = 完全编造或与参考信息矛盾

只输出数字，不要其他文字。"""

    try:
        score = float(judge.ask(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.5


def eval_answer_relevancy(judge: QwenJudge, question: str, answer: str) -> float:
    """评估 Answer Relevancy：回答是否切题"""
    prompt = f"""你是评估助手。请判断以下回答是否与问题相关、切题。

【用户问题】
{question}

【模型回答】
{answer}

请输出一个 0 到 1 之间的分数：
- 1.0 = 完全切题，直接回答
- 0.5 = 部分相关，有偏题
- 0.0 = 完全不相关

只输出数字，不要其他文字。"""

    try:
        score = float(judge.ask(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.5


def eval_context_precision(judge: QwenJudge, question: str, contexts: List[str]) -> float:
    """评估 Context Precision：检索到的上下文质量"""
    if not contexts:
        return 0.0

    context_text = "\n\n---\n\n".join(f"[文档{i+1}]\n{c[:500]}" for i, c in enumerate(contexts))

    prompt = f"""你是评估助手。以下是为用户问题检索到的参考文档。请评估这些文档中有多少比例是真正相关且有用的。

【用户问题】
{question}

【检索到的文档】
{context_text[:4000]}

请输出一个 0 到 1 之间的分数：
- 1.0 = 全部文档都与问题直接相关
- 0.5 = 约一半文档相关
- 0.0 = 没有文档相关

只输出数字，不要其他文字。"""

    try:
        score = float(judge.ask(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.5


# ============================================================
# 主流程
# ============================================================

def evaluate_qa(qa_data: List[Dict], judge: QwenJudge) -> List[Dict]:
    """逐条评估所有 QA"""
    results = []

    for i, qa in enumerate(qa_data):
        # 运行系统
        import asyncio
        from agent.graph import run_agent
        from config import DEFAULT_CONFIG

        async def ask():
            result = await run_agent(query=qa["question"], config=DEFAULT_CONFIG)
            chunks = result.get("retrieved_chunks", [])
            contexts = [c.get("content", "") for c in chunks] if chunks else []
            return {
                "answer": result.get("final_answer", ""),
                "contexts": contexts,
            }

        try:
            sys.stdout.write(f"\r  🔄 [{i+1}/{len(qa_data)}] {qa['question'][:35]}...  ")
            sys.stdout.flush()

            result = asyncio.run(ask())

            # 评分
            faithfulness = eval_faithfulness(judge, qa["question"], result["answer"], result["contexts"])
            relevancy = eval_answer_relevancy(judge, qa["question"], result["answer"])
            precision = eval_context_precision(judge, qa["question"], result["contexts"])

            results.append({
                "id": qa["id"],
                "question": qa["question"],
                "difficulty": qa["difficulty"],
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "context_precision": precision,
                "answer_length": len(result["answer"]),
                "context_count": len(result["contexts"]),
            })

        except Exception as e:
            print(f"\n  ❌ {qa['question'][:30]}: {e}")

    return results


def print_report(results: List[Dict]):
    """输出评估报告"""
    if not results:
        print("\n❌ 无评估结果")
        return

    faith_scores = [r["faithfulness"] for r in results]
    relevancy_scores = [r.get("answer_relevancy", 0) for r in results]
    precision_scores = [r.get("context_precision", 0) for r in results]

    print()
    print("=" * 55)
    print("  RAG 评估报告（本地 Judge: qwen-turbo）")
    print("=" * 55)
    print(f"  评估 QA 数: {len(results)}")
    print()
    print(f"  Faithfulness:       {sum(faith_scores)/len(faith_scores):.3f} (平均)")
    print(f"  Answer Relevancy:   {sum(relevancy_scores)/len(relevancy_scores):.3f} (平均)")
    print(f"  Context Precision:  {sum(precision_scores)/len(precision_scores):.3f} (平均)")
    print()

    # 按难度分层
    print("  按难度分层:")
    for diff in ["L1_factual", "L2_comparison", "L3_reasoning", "L4_cross_doc"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        diff_label = {"L1_factual": "L1 事实性", "L2_comparison": "L2 比较性",
                       "L3_reasoning": "L3 推理", "L4_cross_doc": "L4 跨文档"}.get(diff, diff)
        avg_f = sum(r["faithfulness"] for r in subset) / len(subset)
        avg_p = sum(r.get("context_precision", 0) for r in subset) / len(subset)
        print(f"    {diff_label:>15}: f={avg_f:.3f}, p={avg_p:.3f} ({len(subset)}条)")

    print()
    print("  各条目详情:")
    for r in results:
        print(f"    {r['id']:>8}: f={r['faithfulness']:.2f}, "
              f"r={r.get('answer_relevancy', 0):.2f}, "
              f"p={r.get('context_precision', 0):.2f} "
              f"| {r['question'][:35]}")

    print()
    print(f"  {'='*55}")
    print(f"  报告已保存: data/eval/local_eval_report.json")

    # 保存
    save_path = Path("data/eval/local_eval_report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    print("🔬 本地 RAG 评估（Judge: qwen-turbo）")
    print()

    # 检查 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量")
        return

    # 加载数据
    with open("data/eval/qa_benchmark.json", encoding="utf-8") as f:
        qa_data = json.load(f)

    print(f"📊 加载了 {len(qa_data)} 个 QA pair")
    print()

    # 初始化 judge
    judge = QwenJudge()
    print("✅ Judge LLM (qwen-turbo) 已初始化")

    # 评估
    results = evaluate_qa(qa_data, judge)

    # 报告
    print_report(results)


if __name__ == "__main__":
    main()
