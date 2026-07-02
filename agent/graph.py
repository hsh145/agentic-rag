"""
LangGraph 工作流编排 - 6 节点 + 条件边
"""
import logging

from langgraph.graph import StateGraph, START, END

from .state import AgenticRAGState, create_initial_state
from .nodes import (
    parse_intent, parse_files, plan_retrieval,
    execute_search, evaluate_evidence, generate_answer,
    route_after_intent, route_after_search,
)

logger = logging.getLogger("agent.graph")


def build_agent() -> StateGraph:
    """构建 Agentic RAG 工作流"""

    workflow = StateGraph(AgenticRAGState)

    # ===== 注册 6 个节点 =====
    workflow.add_node("parse_intent", parse_intent)         # ① 意图分析
    workflow.add_node("parse_files", parse_files)           # ② 文件解析（条件执行）
    workflow.add_node("plan_retrieval", plan_retrieval)     # ③ 检索规划
    workflow.add_node("execute_search", execute_search)     # ④ 执行检索
    workflow.add_node("evaluate_evidence", evaluate_evidence)  # ⑤ 证据评估
    workflow.add_node("generate_answer", generate_answer)   # ⑥ 回答生成

    # ===== 编排流程 =====
    workflow.add_edge(START, "parse_intent")

    # 意图分析 → 条件路由
    workflow.add_conditional_edges(
        "parse_intent",
        route_after_intent,
        {
            "parse_files": "parse_files",
            "plan_retrieval": "plan_retrieval",
        },
    )

    # 文件解析 → 检索规划
    workflow.add_edge("parse_files", "plan_retrieval")

    # 检索规划 → 执行检索
    workflow.add_edge("plan_retrieval", "execute_search")

    # 执行检索 → 证据评估
    workflow.add_edge("execute_search", "evaluate_evidence")

    # 证据评估 → 条件路由（不够就补搜，够了就生成）
    workflow.add_conditional_edges(
        "evaluate_evidence",
        route_after_search,
        {
            "plan_retrieval": "plan_retrieval",
            "generate_answer": "generate_answer",
        },
    )

    # 生成 → 结束
    workflow.add_edge("generate_answer", END)

    logger.info("Agentic RAG 工作流构建完成（6节点 + 2条件边）")
    return workflow.compile()


async def run_agent(query: str, file_paths=None, config=None):
    """便捷调用入口"""
    app = build_agent()
    initial = create_initial_state(query, file_paths)
    if config:
        initial["max_iterations"] = getattr(config, "max_iterations", 2)
    result = await app.ainvoke(initial)
    return result
