"""
训练数据生成脚本 — 使用 qwen-max 作为 Teacher 生成蒸馏数据

用法：
    python scripts/generate_training_data.py --count 3000 --output data/training/judge_data.jsonl

输出格式：
  每条记录 = {"instruction": "...", "output": "..."} 适配 LLaMA-Factory
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

# 任务模板 — 覆盖 3 个判断节点 + 检索规划 + 反射补搜
TASK_TEMPLATES = [
    # ---- 意图分析 ----
    {
        "node": "parse_intent",
        "system": "你是一个意图分析专家。分析用户问题，判断是否需要解析文件、检索知识库，以及需求类型。",
        "user_template": "分析用户问题，输出 JSON：\n\n用户问题：{query}\n\n请判断：\n1. 是否需要解析文件\n2. 是否需要检索知识库\n3. 需求类型：simple（简单问答）/ file_process（文件处理）/ deep_search（深度检索）\n\n输出JSON：\n{{\n    \"need_file_parse\": true/false,\n    \"need_rag_search\": true/false,\n    \"query_type\": \"simple|file_process|deep_search\",\n    \"analysis\": \"简要分析\"\n}}",
    },
    # ---- 检索规划 ----
    {
        "node": "plan_retrieval",
        "system": "你是一个检索规划专家。将复杂查询拆解为多个独立的搜索子查询。",
        "user_template": "将以下用户问题拆解为 1-3 个独立的搜索子查询。\n每个子查询应该覆盖问题的一个独立维度。\n\n用户问题: {query}\n\n输出 JSON：\n{{\n    \"sub_queries\": [\"子查询1\", \"子查询2\"],\n    \"reasoning\": \"拆解思路\"\n}}",
    },
    # ---- 证据评估 ----
    {
        "node": "evaluate_evidence",
        "system": "你是一个证据评估专家。判断检索到的信息是否足够回答用户问题。",
        "user_template": "判断当前检索到的信息是否足够回答用户问题。\n\n用户问题：{query}\n\n检索到的 {chunk_count} 条证据：\n{context}\n\n请分析并输出 JSON：\n{{\n    \"can_answer\": true/false,\n    \"missing_gaps\": [\"缺口1\"],\n    \"feedback\": \"评估摘要\",\n    \"confidence\": 0.0-1.0\n}}",
    },
]

# 种子 Query 集合（覆盖常见问题类型）
SEED_QUERIES = [
    # 事实性查询
    "什么是SFT微调？",
    "LoRA的全称是什么？",
    "FAISS是什么？",
    "什么是RLHF？",
    "SFT微调中learning rate的推荐范围是多少？",
    "Batch size在SFT微调中的推荐范围是什么？",
    "什么是Dpo训练？",
    "Chain-of-Thought提示是什么意思？",
    "什么是检索增强生成（RAG）？",
    "LangChain是什么？",
    # 比较性查询
    "LoRA和Full Fine-tuning的主要区别是什么？",
    "对比SFT和RLHF两种训练方法",
    "对比LangChain和LlamaIndex的优缺点",
    "GPT和BERT的区别是什么？",
    "Prompt Engineering和Fine-tuning有什么不同？",
    # 推理性查询
    "如果显存只有24GB，应该选择哪种微调策略？为什么？",
    "为什么RAG在长尾知识场景下效果好于直接生成？",
    "当检索结果不够时，Agent应该怎么做？",
    "怎样评估一个RAG系统的质量？",
]

# 知识库上下文样本（模拟检索结果）
SAMPLE_CONTEXTS = [
    "SFT（Supervised Fine-Tuning，监督式微调）是在预训练语言模型的基础上，使用高质量的标注数据对模型进行有监督训练的过程。SFT微调中learning rate的推荐范围是1e-5到5e-5，通常选择2e-5作为初始值。",
    "LoRA（Low-Rank Adaptation，低秩自适应）是一种参数高效的微调方法。LoRA通过低秩矩阵只更新0.1%~1%的参数，显存占用约12GB（7B模型），训练速度快3-5倍，效果接近Full Fine-tuning。",
    "FAISS（Facebook AI Similarity Search）是Meta开源的高效向量检索库，用于大规模向量数据的相似度搜索。FAISS支持多种索引类型，包括Flat（精确检索）、IVF（倒排索引）、HNSW（图索引）等。",
    "RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）是一种使用人类反馈来优化语言模型的技术。RLHF通常包含三个步骤：SFT微调、训练奖励模型、PPO强化学习优化。",
    "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的混合架构。它首先从知识库中检索相关文档片段，然后将这些片段作为上下文提供给生成模型，以提高回答的准确性和时效性。",
    "Batch size在SFT微调中推荐8-32，根据显存大小调整。太大的Batch size可能导致收敛不稳定，建议小批量+梯度累积的组合策略。",
    "DPO（Direct Preference Optimization）是一种直接优化人类偏好的训练方法，不需要训练奖励模型，比RLHF更简单高效。",
    "Chain-of-Thought（思维链）是一种提示技术，通过在问题后添加中间推理步骤的示例，引导模型逐步推理，而不是直接输出最终答案。",
    "LangChain是一个用于构建LLM应用的框架，提供了链式调用、工具集成、记忆管理等模块化组件。",
    "LlamaIndex是一个专注于数据索引和检索的框架，提供了丰富的文档加载器、索引结构和查询引擎。",
]


def generate_parse_intent_data(teacher_llm, queries: List[str]) -> List[Dict]:
    """生成意图分析训练数据"""
    results = []
    for q in queries:
        prompt = TASK_TEMPLATES[0]["user_template"].format(query=q)
        try:
            response = teacher_llm.invoke([HumanMessage(content=prompt)])
            output = response.content.strip().replace("```json", "").replace("```", "").strip()
            json.loads(output)  # 验证是合法 JSON
            results.append({
                "instruction": prompt,
                "output": output,
                "node": "parse_intent",
                "source_query": q,
            })
        except Exception as e:
            print(f"  ❌ {q[:20]}: {e}")
    return results


def generate_plan_retrieval_data(teacher_llm, queries: List[str]) -> List[Dict]:
    """生成检索规划训练数据"""
    results = []
    for q in queries:
        prompt = TASK_TEMPLATES[1]["user_template"].format(query=q)
        try:
            response = teacher_llm.invoke([HumanMessage(content=prompt)])
            output = response.content.strip().replace("```json", "").replace("```", "").strip()
            json.loads(output)
            results.append({
                "instruction": prompt,
                "output": output,
                "node": "plan_retrieval",
                "source_query": q,
            })
        except Exception as e:
            print(f"  ❌ {q[:20]}: {e}")
    return results


def generate_evaluate_evidence_data(teacher_llm, queries: List[str], contexts: List[str]) -> List[Dict]:
    """生成证据评估训练数据"""
    results = []
    import random
    for q in queries:
        # 随机选 1-3 个上下文
        k = random.randint(1, 3)
        selected = random.sample(contexts, min(k, len(contexts)))
        context_text = "\n\n".join(f"[来源{i+1}]\n{c}" for i, c in enumerate(selected))

        prompt = TASK_TEMPLATES[2]["user_template"].format(
            query=q, chunk_count=len(selected), context=context_text
        )
        try:
            response = teacher_llm.invoke([HumanMessage(content=prompt)])
            output = response.content.strip().replace("```json", "").replace("```", "").strip()
            json.loads(output)
            results.append({
                "instruction": prompt,
                "output": output,
                "node": "evaluate_evidence",
                "source_query": q,
            })
        except Exception as e:
            print(f"  ❌ {q[:20]}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="生成蒸馏训练数据")
    parser.add_argument("--count", type=int, default=500, help="目标数据量")
    parser.add_argument("--output", type=str, default="data/training/judge_data.jsonl", help="输出路径")
    args = parser.parse_args()

    from langchain_core.messages import HumanMessage
    from langchain_community.chat_models import ChatTongyi

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("❌ 请设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    # Teacher 模型：qwen-max
    teacher = ChatTongyi(model="qwen-max", temperature=0.0, dashscope_api_key=api_key)
    print(f"✅ Teacher: qwen-max 已连接")
    print(f"🎯 目标数据量: {args.count} 条")

    all_data = []

    # 扩展种子 query（通过随机组合生成更多变体）
    queries = SEED_QUERIES.copy()
    import random
    random.seed(42)
    while len(queries) < args.count // 2:
        q = random.choice(SEED_QUERIES)
        queries.append(q)

    print(f"\n📝 生成意图分析数据...")
    data1 = generate_parse_intent_data(teacher, queries[:args.count // 3])
    all_data.extend(data1)
    print(f"   ✅ {len(data1)} 条")

    print(f"\n📝 生成检索规划数据...")
    data2 = generate_plan_retrieval_data(teacher, queries[args.count // 3: 2 * args.count // 3])
    all_data.extend(data2)
    print(f"   ✅ {len(data2)} 条")

    print(f"\n📝 生成证据评估数据...")
    data3 = generate_evaluate_evidence_data(teacher, queries[2 * args.count // 3: args.count], SAMPLE_CONTEXTS)
    all_data.extend(data3)
    print(f"   ✅ {len(data3)} 条")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"✅ 共生成 {len(all_data)} 条训练数据")
    print(f"📁 保存到: {output_path}")
    print(f"   意图分析: {len(data1)} 条")
    print(f"   检索规划: {len(data2)} 条")
    print(f"   证据评估: {len(data3)} 条")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
