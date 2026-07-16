"""
训练数据转换器 — 将 generate_training_data.py 的输出转为 LLaMA-Factory sharegpt 格式

用法：
    python scripts/convert_to_llamafactory.py --input data/training/judge_data.jsonl --output data/training/judge_data_sharegpt.jsonl
"""

import json
import argparse
from pathlib import Path

# 每个节点类型对应的系统提示
SYSTEM_PROMPTS = {
    "parse_intent": "你是一个意图分析专家。分析用户问题，判断是否需要解析文件、检索知识库，以及需求类型。输出 JSON 格式的结果。",
    "plan_retrieval": "你是一个检索规划专家。将复杂查询拆解为多个独立的搜索子查询。输出 JSON 格式的结果。",
    "evaluate_evidence": "你是一个证据评估专家。判断检索到的信息是否足够回答用户问题。输出 JSON 格式的结果。",
}


def convert(input_path: str, output_path: str):
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line.strip())
            node = item.get("node", "parse_intent")
            system = SYSTEM_PROMPTS.get(node, "")

            sharegpt = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]},
                ],
                "node": node,
            }
            fout.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ 转换完成: {count} 条")
    print(f"📁 输入: {input_path}")
    print(f"📁 输出: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/judge_data.jsonl")
    parser.add_argument("--output", default="data/training/judge_data_sharegpt.jsonl")
    args = parser.parse_args()
    convert(args.input, args.output)
