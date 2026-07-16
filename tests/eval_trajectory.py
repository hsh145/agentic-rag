"""
Agentic 轨迹评估 — 停止决策质量 / 补搜有效性 / 轮次效率

不是只看"最终答案对不对"，而是追踪 agent 每跳的决策质量：

  1. 停止决策质量：
     evaluate_evidence 判断"够了/不够"这个二分类的准确率。
     统计"信息不够但模型说够了"（过早停止）和"信息够了但还在补搜"（过晚停止）。

  2. 补搜有效性：
     reflect_search 生成的新 query 有多少比例真的检索到了与首轮不重复的新内容。
     衡量边际收益：第 N 轮新增的独特 chunk 数。

  3. 轮次效率：
     平均迭代次数、每次迭代的 chunk 增量分布。

用法：
    python tests/eval_trajectory.py

输出：
    - 控制台报告
    - data/eval/trajectory_report.json
"""
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# 数据模型
# ============================================================

@dataclass
class HopRecord:
    """单跳记录"""
    hop: int
    query: str                          # 本轮使用的查询
    query_type: str                     # "initial" | "supplementary"
    chunk_count: int                    # 本轮检索到的 chunk 数
    new_chunk_count: int                # 本轮新增（去重后）的 chunk 数
    unique_chunk_ids: List[int] = field(default_factory=list)  # 本轮新增 chunk 的索引

@dataclass
class TrajectoryResult:
    """一条 QA 的完整轨迹评估结果"""
    qa_id: str
    question: str
    difficulty: str
    n_hops: int                         # 实际迭代次数
    stop_decision: str                  # "correct_stop" | "early_stop" | "late_stop"
    stop_confidence: float              # evaluate_evidence 给出的置信度
    hops: List[HopRecord] = field(default_factory=list)
    all_chunk_ids: set = field(default_factory=set)
    total_unique_chunks: int = 0
    marginal_gains: List[float] = field(default_factory=list)  # 每轮新增比例


# ============================================================
# 模拟 Agentic 循环（不依赖完整 API）
# ============================================================

def simulate_agentic_loop(
    question: str,
    retriever,
    chunk_index: List[Dict],
    relevant_set: set,
    max_iterations: int = 3,
    llm_judge=None,
) -> TrajectoryResult:
    """模拟 Agentic RAG 的"检索→评估→补搜"闭环

    不调真实 LLM（避免成本），用规则模拟 evaluate_evidence 和 reflect_search。
    如果要测真实 LLM 的停止决策质量，设置 llm_judge 为一个 callable。

    Args:
        question: 用户问题
        retriever: HybridRetriever 实例
        chunk_index: chunk 索引表
        relevant_set: 相关 chunk 的索引集合
        max_iterations: 最大迭代次数
        llm_judge: 可选的 LLM judge 函数，接收 (query, chunks) 返回 (can_answer, gaps)

    Returns:
        TrajectoryResult
    """
    result = TrajectoryResult(
        qa_id="",
        question=question,
        difficulty="",
        n_hops=0,
        stop_decision="unknown",
        stop_confidence=0.0,
    )

    # 第 1 跳：原始查询
    docs = retriever.hybrid_search(question, top_k=5)
    hop1_chunks = set()
    hop1_ids = []
    for doc in docs:
        content = doc.page_content[:200]
        for ci in chunk_index:
            if ci["content"] == content and ci["index"] not in hop1_chunks:
                hop1_chunks.add(ci["index"])
                hop1_ids.append(ci["index"])
                break

    result.all_chunk_ids.update(hop1_ids)
    result.hops.append(HopRecord(
        hop=1, query=question, query_type="initial",
        chunk_count=len(hop1_ids), new_chunk_count=len(hop1_ids),
        unique_chunk_ids=hop1_ids,
    ))

    # 模拟 evaluate_evidence（规则版）
    for iteration in range(2, max_iterations + 2):
        current_chunks = result.all_chunk_ids
        relevant_hit = len(current_chunks & relevant_set) if relevant_set else 0
        total_chars = sum(len(chunk_index[i]["content"]) for i in current_chunks if i < len(chunk_index))

        # 停止条件模拟：
        # - 已命中 1+ 个相关 chunk 且总字符数 > 300 → 可以停
        # - 或已迭代 2 次且每次新增 < 20% → 边际收益递减，可以停
        can_answer = False
        stop_reason = ""

        # 条件 A：已有足够相关信息
        if relevant_hit >= 2 and total_chars > 300:
            can_answer = True
            stop_reason = "sufficient_evidence"
            result.stop_confidence = 0.9
        elif relevant_hit >= 1 and total_chars > 500:
            can_answer = True
            stop_reason = "sufficient_evidence"
            result.stop_confidence = 0.8

        # 条件 B：边际收益递减
        if not can_answer and len(result.hops) >= 2:
            prev_new = result.hops[-1].new_chunk_count
            if prev_new < 0.2 * result.hops[-2].new_chunk_count:
                can_answer = True
                stop_reason = "diminishing_returns"
                result.stop_confidence = 0.6

        # 条件 C：达到最大迭代次数
        if iteration > max_iterations:
            can_answer = True
            stop_reason = "max_iterations"
            result.stop_confidence = 0.3

        if can_answer:
            result.n_hops = len(result.hops)
            # 判断停止决策质量
            if relevant_hit == 0:
                result.stop_decision = "early_stop"  # 没命中相关 chunk 就停了
            elif relevant_hit >= len(relevant_set) if len(relevant_set) <= 3 else relevant_hit >= 2:
                result.stop_decision = "correct_stop"  # 命中了大部分相关 chunk
            else:
                result.stop_decision = "early_stop"  # 还有一些相关 chunk 没找到
            break

        # 生成补搜查询（模拟 reflect_search）
        supp_query = f"{question} 详细信息"

        # 执行补搜
        supp_docs = retriever.hybrid_search(supp_query, top_k=5)
        new_ids = []
        for doc in supp_docs:
            content = doc.page_content[:200]
            for ci in chunk_index:
                if ci["content"] == content and ci["index"] not in result.all_chunk_ids:
                    new_ids.append(ci["index"])
                    result.all_chunk_ids.add(ci["index"])
                    break

        result.hops.append(HopRecord(
            hop=iteration, query=supp_query, query_type="supplementary",
            chunk_count=len(supp_docs),
            new_chunk_count=len(new_ids),
            unique_chunk_ids=new_ids,
        ))

        # 如果补搜没找到任何新 chunk → 提前结束
        if not new_ids:
            result.n_hops = len(result.hops)
            result.stop_decision = "correct_stop"
            result.stop_confidence = 0.7
            break

    if result.n_hops == 0:
        result.n_hops = len(result.hops)

    result.total_unique_chunks = len(result.all_chunk_ids)
    result.marginal_gains = [
        h.new_chunk_count / max(result.total_unique_chunks, 1)
        for h in result.hops
    ]

    return result


# ============================================================
# 指标计算
# ============================================================

def eval_stop_decision_accuracy(results: List[TrajectoryResult]) -> Dict:
    """统计停止决策的准确率"""
    total = len(results)
    correct = sum(1 for r in results if r.stop_decision == "correct_stop")
    early = sum(1 for r in results if r.stop_decision == "early_stop")
    late = sum(1 for r in results if r.stop_decision == "late_stop")
    return {
        "total": total,
        "correct_stop": correct,
        "early_stop": early,
        "late_stop": late,
        "correct_rate": correct / total if total else 0,
        "early_stop_rate": early / total if total else 0,
    }


def eval_supplementary_effectiveness(results: List[TrajectoryResult]) -> Dict:
    """评估补搜有效性"""
    new_chunk_counts = []
    supplement_ratios = []

    for r in results:
        if len(r.hops) > 1:
            # 后续轮次新增的 chunk 数 / 总 chunk 数
            supp_total = sum(h.new_chunk_count for h in r.hops[1:])
            new_chunk_counts.append(supp_total)
            supplement_ratios.append(supp_total / max(r.total_unique_chunks, 1))

    return {
        "total_results": len(results),
        "had_supplementary": len(new_chunk_counts),
        "avg_new_chunks_per_supp": sum(new_chunk_counts) / len(new_chunk_counts) if new_chunk_counts else 0,
        "avg_supplement_ratio": sum(supplement_ratios) / len(supplement_ratios) if supplement_ratios else 0,
        "max_new_chunks": max(new_chunk_counts) if new_chunk_counts else 0,
    }


def eval_iteration_efficiency(results: List[TrajectoryResult]) -> Dict:
    """评估轮次效率"""
    hop_counts = [r.n_hops for r in results]
    return {
        "avg_hops": sum(hop_counts) / len(hop_counts) if hop_counts else 0,
        "min_hops": min(hop_counts) if hop_counts else 0,
        "max_hops": max(hop_counts) if hop_counts else 0,
        "one_hop_ratio": sum(1 for h in hop_counts if h == 1) / len(hop_counts) if hop_counts else 0,
        "three_plus_hop_ratio": sum(1 for h in hop_counts if h >= 3) / len(hop_counts) if hop_counts else 0,
    }


# ============================================================
# 主流程
# ============================================================

def run_evaluation():
    print("=" * 60)
    print("  🎯 Agentic 轨迹评估")
    print("  — 停止决策质量 / 补搜有效性 / 轮次效率")
    print("=" * 60)
    print()

    # ---- 构建索引（复用检索层评估的索引构建）----
    from tests.eval_retrieval import build_benchmark_index, load_qa_benchmark

    print("📦 构建 benchmark 索引...")
    t0 = time.time()
    retriever, chunk_index, chunks = build_benchmark_index()
    print(f"  索引就绪（{time.time() - t0:.1f}s）")
    print()

    # ---- 加载 QA ----
    qa_data = load_qa_benchmark()
    print(f"📊 加载了 {len(qa_data)} 个 QA pair")
    print()

    # ---- 建立 source_file → chunk_index 映射 ----
    file_to_chunks_map = {}
    for ci in chunk_index:
        fn = ci["source_file"]
        if fn not in file_to_chunks_map:
            file_to_chunks_map[fn] = set()
        file_to_chunks_map[fn].add(ci["index"])

    # ---- 逐条评估 ----
    results: List[TrajectoryResult] = []

    for qa in qa_data:
        qid = qa["id"]
        question = qa["question"]
        relevant_docs = qa.get("relevant_docs", [])

        # 找到相关 chunk
        relevant_set = set()
        for rd in relevant_docs:
            for fn, chunk_set in file_to_chunks_map.items():
                if fn == rd or fn.endswith(rd) or rd.endswith(fn):
                    relevant_set.update(chunk_set)

        sys.stdout.write(f"\r  🔄 [{qa_data.index(qa)+1}/{len(qa_data)}] {qid} {question[:30]}...  ")
        sys.stdout.flush()

        traj = simulate_agentic_loop(
            question=question,
            retriever=retriever,
            chunk_index=chunk_index,
            relevant_set=relevant_set,
            max_iterations=3,
        )
        traj.qa_id = qid
        traj.question = question
        traj.difficulty = qa.get("difficulty", "unknown")
        results.append(traj)

    print("\n")
    print("=" * 60)
    print("  📊 Agentic 轨迹评估报告")
    print("=" * 60)
    print(f"  评估条目: {len(results)}")

    # ---- 1. 停止决策质量 ----
    stop_stats = eval_stop_decision_accuracy(results)
    print()
    print(f"  1️⃣ 停止决策质量")
    print(f"  {'-'*40}")
    print(f"  正确停止: {stop_stats['correct_stop']} / {stop_stats['total']} ({stop_stats['correct_rate']:.1%})")
    print(f"  过早停止: {stop_stats['early_stop']} / {stop_stats['total']} ({stop_stats['early_stop_rate']:.1%})")
    print(f"  过晚停止: {stop_stats['late_stop']} / {stop_stats['total']}")

    # ---- 2. 补搜有效性 ----
    supp_stats = eval_supplementary_effectiveness(results)
    print()
    print(f"  2️⃣ 补搜有效性")
    print(f"  {'-'*40}")
    print(f"  发生补搜的比例: {supp_stats['had_supplementary']} / {supp_stats['total_results']} ({supp_stats['had_supplementary']/max(supp_stats['total_results'],1):.1%})")
    print(f"  补搜平均新增 chunk: {supp_stats['avg_new_chunks_per_supp']:.1f}")
    print(f"  补搜新增占比: {supp_stats['avg_supplement_ratio']:.1%}")

    # ---- 3. 轮次效率 ----
    iter_stats = eval_iteration_efficiency(results)
    print()
    print(f"  3️⃣ 轮次效率")
    print(f"  {'-'*40}")
    print(f"  平均迭代次数: {iter_stats['avg_hops']:.2f}")
    print(f"  迭代范围: {iter_stats['min_hops']} ~ {iter_stats['max_hops']}")
    print(f"  单轮解决: {iter_stats['one_hop_ratio']:.1%}")
    print(f"  三轮以上: {iter_stats['three_plus_hop_ratio']:.1%}")

    # ---- 按难度分层 ----
    print()
    print(f"  4️⃣ 按难度分层")
    print(f"  {'难度':>12}  {'条目':>4}  {'平均轮次':>8}  {'正确停止率':>10}  {'补搜新增':>8}")
    print(f"  {'-'*12}  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}")
    for diff in ["L1_factual", "L2_comparison", "L3_reasoning", "L4_cross_doc"]:
        subset = [r for r in results if r.difficulty == diff]
        if not subset:
            continue
        diff_label = {"L1_factual": "L1 事实性", "L2_comparison": "L2 比较性",
                      "L3_reasoning": "L3 推理", "L4_cross_doc": "L4 跨文档"}.get(diff, diff)
        avg_h = sum(r.n_hops for r in subset) / len(subset)
        correct_r = sum(1 for r in subset if r.stop_decision == "correct_stop") / len(subset)
        avg_supp = sum(
            sum(h.new_chunk_count for h in r.hops[1:]) for r in subset
        ) / len(subset)
        print(f"  {diff_label:>12}  {len(subset):>4}  {avg_h:.2f}{'':>6}  {correct_r:.1%}{'':>7}  {avg_supp:.1f}")

    # ---- 保存 ----
    save_path = ROOT / "data" / "eval" / "trajectory_report.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "max_iterations": 3,
            "evaluator": "rule_based",
        },
        "stop_decision": stop_stats,
        "supplementary": supp_stats,
        "iteration_efficiency": iter_stats,
        "per_qa": [
            {
                "id": r.qa_id,
                "question": r.question,
                "difficulty": r.difficulty,
                "n_hops": r.n_hops,
                "stop_decision": r.stop_decision,
                "total_unique_chunks": r.total_unique_chunks,
                "marginal_gains": r.marginal_gains,
                "hops": [asdict(h) for h in r.hops],
            }
            for r in results
        ],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"  💾 报告已保存: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
