"""
Golden Set 回归测试 — 7 类问题的全链路评估

覆盖类型：
  factual      事实问题（单片段可答）
  method       方法问题（解释机制/流程）
  comparison   对比问题（跨片段/文档比较）
  summary      总结问题（综合多个证据）
  ambiguous    模糊问题（需要澄清或保守回答）
  no_evidence  无证据问题（应拒答）
  false_premise 错误前提问题（应纠正前提）

用法：
    # 需要后端在运行
    python tests/eval_golden.py

    或指定后端地址：
    python tests/eval_golden.py --url http://localhost:8000

输出：
    - data/eval/golden_report.json（结构化结果）
    - 控制台按类别输出指标
"""
import json
import sys
import time
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# 数据模型
# ============================================================

@dataclass
class GoldenItem:
    """一条 golden 测试样本"""
    id: str
    question: str
    category: str                     # factual|method|comparison|summary|ambiguous|no_evidence|false_premise
    difficulty: str                   # easy|medium|hard
    expected_docs: List[str] = field(default_factory=list)
    expected_chunks: List[str] = field(default_factory=list)
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    answer_notes: str = ""
    ground_truth: str = ""
    expect_clarification: bool = False
    expect_refusal: bool = False
    expect_premise_correction: bool = False


@dataclass
class GoldenResult:
    """一条测试结果"""
    id: str
    category: str
    difficulty: str
    question: str
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    retrieved_chunks: List[str] = field(default_factory=list)
    iterations: int = 0

    # 检索指标
    doc_recall: float = 0.0           # 期望文档被召回的比率
    chunk_recall: float = 0.0         # 期望 chunk 主题被覆盖的比率

    # 生成指标
    must_contain_recall: float = 0.0  # must_contain 命中率
    must_not_contain_violation: bool = False  # 是否出现了违禁词
    refusal_correct: bool = False     # no_evidence → 是否正确拒答
    clarification_correct: bool = False  # ambiguous → 是否澄清
    premise_correct: bool = False     # false_premise → 是否纠正前提

    # 期望行为（从 GoldenItem 复制过来，用于统计）
    expect_refusal: bool = False
    expect_clarification: bool = False
    expect_premise_correction: bool = False

    # 综合
    passed: bool = False
    errors: List[str] = field(default_factory=list)


# ============================================================
# 加载 golden set
# ============================================================

def load_golden_set(path: Path = None) -> List[GoldenItem]:
    if path is None:
        path = ROOT / "data" / "eval" / "golden_set.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for d in data:
        items.append(GoldenItem(
            id=d["id"],
            question=d["question"],
            category=d.get("category", "unknown"),
            difficulty=d.get("difficulty", "medium"),
            expected_docs=d.get("expected_docs", []),
            expected_chunks=d.get("expected_chunks", []),
            must_contain=d.get("must_contain", []),
            must_not_contain=d.get("must_not_contain", []),
            answer_notes=d.get("answer_notes", ""),
            ground_truth=d.get("ground_truth", ""),
            expect_clarification=d.get("expect_clarification", False),
            expect_refusal=d.get("expect_refusal", False),
            expect_premise_correction=d.get("expect_premise_correction", False),
        ))
    return items


# ============================================================
# 调用后端 API
# ============================================================

API_URL = "http://localhost:8000"


def call_api(question: str, url: str = API_URL) -> Dict:
    """通过 /api/ask/trace 获取完整回答和轨迹"""
    payload = {
        "query": question,
        "file_paths": [],
        "max_iterations": 2,
        "session_id": f"golden_{int(time.time())}",
    }
    try:
        resp = requests.post(
            f"{url}/api/ask/trace",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"无法连接到后端 {url}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 检查函数
# ============================================================

def check_keywords(text: str, keywords: List[str]) -> float:
    """检查文本中关键词的命中率"""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def has_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def check_refusal(text: str) -> bool:
    """检查回答是否表达了无法回答/信息不足"""
    refusal_patterns = [
        "没有相关信息", "无法回答", "信息不足", "不知道",
        "没有找到", "无法提供", "不在知识库", "没有收录",
        "我无法", "无法确认", "超出", "暂无",
    ]
    return any(p in text for p in refusal_patterns)


def check_clarification(text: str) -> bool:
    """检查回答是否要求澄清"""
    clarify_patterns = [
        "请提供", "请明确", "具体是哪个", "没有指代",
        "指的是", "哪个方面", "能具体说明一下吗",
        "请补充", "不太清楚您指的是",
    ]
    return any(p in text for p in clarify_patterns)


def check_premise_correction(text: str) -> bool:
    """检查回答是否纠正了错误前提"""
    correction_patterns = [
        "实际上", "前提不正确", "前提有误", "其实",
        "需要纠正", "理解有误", "不对", "并非",
        "误区", "常见的误解", "正确的理解是",
        "需要澄清", "前提错误",
    ]
    return any(p in text for p in correction_patterns)


# ============================================================
# 评估单条
# ============================================================

def evaluate_item(item: GoldenItem, api_result: Dict) -> GoldenResult:
    """对一条 golden 样本执行评估"""
    result = GoldenResult(
        id=item.id,
        category=item.category,
        difficulty=item.difficulty,
        question=item.question,
        expect_refusal=item.expect_refusal,
        expect_clarification=item.expect_clarification,
        expect_premise_correction=item.expect_premise_correction,
    )

    if not api_result.get("success"):
        result.errors.append(api_result.get("error", "API 调用失败"))
        return result

    answer = api_result.get("answer", "")
    sources = api_result.get("sources", [])
    retrieved_chunks = [
        c.get("content_snippet", "") for c in api_result.get("retrieved_chunks", [])
    ]
    result.answer = answer
    result.sources = sources
    result.retrieved_chunks = retrieved_chunks
    result.iterations = api_result.get("iterations", 0)

    # ---- 检索召回 ----
    if item.expected_docs:
        source_names = " ".join(s.lower() for s in sources)
        doc_hits = sum(
            1 for doc in item.expected_docs
            if doc.lower() in source_names
        )
        result.doc_recall = doc_hits / len(item.expected_docs)

    if item.expected_chunks:
        all_text = " ".join(retrieved_chunks).lower()
        chunk_hits = sum(
            1 for topic in item.expected_chunks
            if topic.lower() in all_text
        )
        result.chunk_recall = chunk_hits / len(item.expected_chunks)

    # ---- 关键词覆盖率 ----
    result.must_contain_recall = check_keywords(answer, item.must_contain)
    result.must_not_contain_violation = any(
        kw.lower() in answer.lower() for kw in item.must_not_contain
    )

    # ---- 特殊类型判断 ----
    if item.expect_refusal:
        result.refusal_correct = check_refusal(answer)

    if item.expect_clarification:
        result.clarification_correct = check_clarification(answer)

    if item.expect_premise_correction:
        result.premise_correct = check_premise_correction(answer)

    # ---- 综合通过判定 ----
    fail_reasons = []

    if item.must_contain and result.must_contain_recall < 0.5:
        fail_reasons.append(f"关键词覆盖率不足 ({result.must_contain_recall:.0%})")
    if result.must_not_contain_violation and item.must_not_contain:
        fail_reasons.append("包含违禁词")

    if item.expect_refusal and not result.refusal_correct:
        fail_reasons.append("应拒答但未拒答")
    if item.expect_clarification and not result.clarification_correct:
        fail_reasons.append("应澄清但未澄清")
    if item.expect_premise_correction and not result.premise_correct:
        fail_reasons.append("应纠正前提但未纠正")

    if item.expected_docs and result.doc_recall == 0:
        fail_reasons.append("未召回任何期望文档")

    result.passed = len(fail_reasons) == 0
    result.errors = fail_reasons
    return result


# ============================================================
# 报告汇总
# ============================================================

def generate_report(results: List[GoldenResult]) -> Dict:
    """生成结构化报告"""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    categories = {}

    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "items": []}
        categories[cat]["total"] += 1
        categories[cat]["items"].append({
            "id": r.id,
            "question": r.question,
            "passed": r.passed,
            "doc_recall": r.doc_recall,
            "chunk_recall": r.chunk_recall,
            "must_contain_recall": r.must_contain_recall,
            "must_not_contain_violation": r.must_not_contain_violation,
            "errors": r.errors,
        })
        if r.passed:
            categories[cat]["passed"] += 1

    per_category = {}
    for cat, data in categories.items():
        per_category[cat] = {
            "pass_rate": data["passed"] / data["total"] if data["total"] else 0,
            "passed": data["passed"],
            "total": data["total"],
        }

    # 总体指标
    avg_doc_recall = sum(r.doc_recall for r in results) / total if total else 0
    avg_chunk_recall = sum(r.chunk_recall for r in results) / total if total else 0
    avg_must_contain = sum(
        r.must_contain_recall for r in results if r.must_contain_recall > 0
    ) / max(sum(1 for r in results if r.must_contain_recall > 0), 1)

    no_evidence_hits = sum(1 for r in results if r.refusal_correct)
    no_evidence_total = sum(1 for r in results if r.expect_refusal)
    false_premise_hits = sum(1 for r in results if r.premise_correct)
    false_premise_total = sum(1 for r in results if r.expect_premise_correction)
    ambiguous_hits = sum(1 for r in results if r.clarification_correct)
    ambiguous_total = sum(1 for r in results if r.expect_clarification)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0,
            "avg_doc_recall": round(avg_doc_recall, 4),
            "avg_chunk_recall": round(avg_chunk_recall, 4),
            "avg_must_contain_recall": round(avg_must_contain, 4),
            "refusal_accuracy": round(no_evidence_hits / max(no_evidence_total, 1), 4),
            "premise_correction_rate": round(false_premise_hits / max(false_premise_total, 1), 4),
            "clarification_rate": round(ambiguous_hits / max(ambiguous_total, 1), 4),
        },
        "per_category": per_category,
        "results": [
            {
                "id": r.id,
                "category": r.category,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "doc_recall": r.doc_recall,
                "chunk_recall": r.chunk_recall,
                "must_contain_recall": r.must_contain_recall,
                "must_not_contain_violation": r.must_not_contain_violation,
                "refusal_correct": r.refusal_correct,
                "premise_correct": r.premise_correct,
                "clarification_correct": r.clarification_correct,
                "question": r.question,
                "errors": r.errors,
                "iterations": r.iterations,
                "answer_snippet": r.answer[:200] if r.answer else "",
            }
            for r in results
        ],
    }


def print_report(report: Dict):
    """打印控制台报告"""
    s = report["summary"]
    print()
    print("=" * 60)
    print("  📋 Golden Set 回归测试报告")
    print("=" * 60)
    print(f"  总计: {s['total']} 条")
    print(f"  通过: {s['passed']} 条 ({s['pass_rate']:.1%})")
    print()
    print(f"  {'指标':>24}  {'分数':>8}")
    print(f"  {'-'*24}  {'-'*8}")
    print(f"  {'文档召回率 (Doc Recall)':>24}  {s['avg_doc_recall']:.2%}")
    print(f"  {'Chunk 主题覆盖':>24}  {s['avg_chunk_recall']:.2%}")
    print(f"  {'关键词覆盖率 (Must-Contain)':>24}  {s['avg_must_contain_recall']:.2%}")
    print(f"  {'无证据拒答准确率':>24}  {s['refusal_accuracy']:.1%}")
    print(f"  {'错误前提纠正率':>24}  {s['premise_correction_rate']:.1%}")
    print(f"  {'模糊问题澄清率':>24}  {s['clarification_rate']:.1%}")
    print()

    # 按类别
    print(f"  {'类别':>16}  {'通过/总数':>10}  {'通过率':>8}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*8}")
    cat_labels = {
        "factual": "事实", "method": "方法",
        "comparison": "对比", "summary": "总结",
        "ambiguous": "模糊", "no_evidence": "无证据",
        "false_premise": "错误前提",
    }
    for cat, data in report["per_category"].items():
        label = cat_labels.get(cat, cat)
        print(f"  {label:>16}  {data['passed']}/{data['total']}{'':>5}  {data['pass_rate']:.0%}")

    print()

    # 失败详情
    failed = [r for r in report["results"] if not r["passed"]]
    if failed:
        print(f"  ❌ 失败条目 ({len(failed)}):")
        for r in failed:
            err_str = "; ".join(r["errors"][:3])
            print(f"    {r['id']} [{r['category']}] {r['question'][:40]}...")
            print(f"      → {err_str}")
        print()

    print("=" * 60)


# ============================================================
# 主流程
# ============================================================

def run_evaluation(url: str = API_URL):
    print("=" * 60)
    print("  🧪 Golden Set 回归测试")
    print("  覆盖 7 类问题：事实 / 方法 / 对比 / 总结 / 模糊 / 无证据 / 错误前提")
    print("=" * 60)
    print()

    # 加载 golden set
    golden = load_golden_set()
    print(f"📋 加载了 {len(golden)} 条 golden 样本")

    # 按类别统计
    cat_counts = Counter(item.category for item in golden)
    for cat, cnt in cat_counts.most_common():
        print(f"   - {cat}: {cnt} 条")
    print()

    # 逐条评估
    results: List[GoldenResult] = []
    for i, item in enumerate(golden):
        label = f"[{i+1}/{len(golden)}] {item.id} ({item.category})"
        sys.stdout.write(f"\r  🔄 {label} {item.question[:40]}...  ")
        sys.stdout.flush()

        t0 = time.time()
        api_result = call_api(item.question, url)
        elapsed = time.time() - t0

        result = evaluate_item(item, api_result)
        status = "✅" if result.passed else "❌"
        print(f"\r  {status} {label} ({elapsed:.0f}s)")

        if result.errors:
            for err in result.errors:
                print(f"     ⚠️  {err}")

        results.append(result)

    # 生成报告
    print()
    report = generate_report(results)

    save_path = ROOT / "data" / "eval" / "golden_report.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"  💾 报告已保存: {save_path}")
    print()


if __name__ == "__main__":
    url = API_URL
    if len(sys.argv) > 1 and sys.argv[1] == "--url" and len(sys.argv) > 2:
        url = sys.argv[2]
    run_evaluation(url)
