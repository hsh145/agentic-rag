#!/usr/bin/env python
"""
Agentic RAG 全量评估 — 顺序执行四层评估并汇总报告

用法：
    python scripts/run_all_evals.py

输出：
    - data/eval/retrieval_report.json    （第1层：检索层）
    - data/eval/ragas_result.csv         （第2层：生成层）
    - data/eval/trajectory_report.json   （第3层：Agentic轨迹）
    - data/eval/summary_report.md        （汇总报告）
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def step_retrieval():
    """第1层：检索层评估"""
    print_header("第1层：检索层 — Recall@k / MRR / Precision@k")
    from tests.eval_retrieval import run_evaluation
    run_evaluation()
    # 加载报告中的摘要
    report_path = ROOT / "data" / "eval" / "retrieval_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f).get("summary", {})
    return {}


def step_generation():
    """第2层：生成层评估 — RAGAS"""
    print_header("第2层：生成层 — RAGAS Faithfulness / Relevancy / Recall / Precision")
    from tests.eval_ragas_local import main as ragas_main
    try:
        ragas_main()
    except Exception as e:
        print(f"  ⚠️  RAGAS 评估失败: {e}")
        print(f"  提示: 需要 DASHSCOPE_API_KEY 和已安装 ragas")
    # 加载报告
    report_path = ROOT / "data" / "eval" / "local_eval_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return []


def step_trajectory():
    """第3层：Agentic轨迹评估"""
    print_header("第3层：Agentic轨迹 — 停止决策 / 补搜有效性 / 轮次效率")
    from tests.eval_trajectory import run_evaluation as traj_eval
    traj_eval()
    report_path = ROOT / "data" / "eval" / "trajectory_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {}


def generate_summary(retrieval_summary, trajectory_report):
    """生成汇总报告"""
    lines = []
    lines.append("# Agentic RAG 评估汇总报告")
    lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---")

    # 第1层
    lines.append("\n## 第1层：检索层")
    if retrieval_summary:
        lines.append(f"\n- 评估条目: {retrieval_summary.get('evaluated', 'N/A')}")
        lines.append(f"- Recall@3: **{retrieval_summary.get('avg_recall@3', 'N/A')}**")
        lines.append(f"- Precision@3: **{retrieval_summary.get('avg_precision@3', 'N/A')}**")
        lines.append(f"- MRR: **{retrieval_summary.get('avg_mrr', 'N/A')}**")

    # 第2层
    lines.append("\n## 第2层：生成层（RAGAS）")
    lines.append("\n```bash\n# 运行：\npython tests/eval_ragas_local.py\n```")

    # 第3层
    lines.append("\n## 第3层：Agentic轨迹")
    if trajectory_report:
        sd = trajectory_report.get("stop_decision", {})
        lines.append(f"\n- 正确停止率: **{sd.get('correct_rate', 'N/A'):.1%}**")
        lines.append(f"- 过早停止率: **{sd.get('early_stop_rate', 'N/A'):.1%}**")
        ie = trajectory_report.get("iteration_efficiency", {})
        lines.append(f"- 平均迭代次数: **{ie.get('avg_hops', 'N/A'):.2f}**")
        lines.append(f"- 单轮解决率: **{ie.get('one_hop_ratio', 'N/A'):.1%}**")

    # 第4层
    lines.append("\n## 第4层：系统性能")
    lines.append("\n```bash\n# 运行：\npython scripts/run_benchmark.py\n# 并发压测：\nlocust -f scripts/locustfile.py\n```")

    lines.append("\n---")
    lines.append("\n*报告由 scripts/run_all_evals.py 自动生成*")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  🧪 Agentic RAG 全量评估")
    print("=" * 60)
    t_start = time.time()

    # 第1层
    retrieval_summary = step_retrieval()

    # 第2层
    step_generation()

    # 第3层
    trajectory_report = step_trajectory()

    # 汇总
    print_header("生成汇总报告")
    summary = generate_summary(retrieval_summary, trajectory_report)
    summary_path = ROOT / "data" / "eval" / "summary_report.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    elapsed = time.time() - t_start
    print(f"\n  ✅ 全量评估完成（{elapsed:.0f}s）")
    print(f"  📄 汇总报告: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
