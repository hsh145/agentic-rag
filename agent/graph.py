"""
LangGraph 工作流编排 - 6 节点 + 条件边 + Memory Checkpointer
"""
import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgenticRAGState, create_initial_state
from .nodes import (
    parse_intent, parse_files, plan_retrieval,
    execute_search, evaluate_evidence, reflect_search,
    generate_answer,
    route_after_intent, route_after_search,
)

logger = logging.getLogger("agent.graph")


def build_agent(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """构建 Agentic RAG 工作流

    Args:
        checkpointer: LangGraph Checkpointer，用于持久化图状态。
                      传 None 则无状态（默认行为）。

    Returns:
        编译后的 StateGraph
    """

    workflow = StateGraph(AgenticRAGState)

    # ===== 注册 7 个节点 =====
    workflow.add_node("parse_intent", parse_intent)             # ① 意图分析
    workflow.add_node("parse_files", parse_files)               # ② 文件解析（条件执行）
    workflow.add_node("plan_retrieval", plan_retrieval)         # ③ 检索规划
    workflow.add_node("execute_search", execute_search)         # ④ 执行检索
    workflow.add_node("evaluate_evidence", evaluate_evidence)   # ⑤ LLM证据评估
    workflow.add_node("reflect_search", reflect_search)         # ⑤b 反射补搜（新增）
    workflow.add_node("generate_answer", generate_answer)       # ⑥ 回答生成

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

    # 证据评估 → 条件路由（LLM判断不够→反射补搜，够了→生成）
    workflow.add_conditional_edges(
        "evaluate_evidence",
        route_after_search,
        {
            "reflect_search": "reflect_search",
            "generate_answer": "generate_answer",
        },
    )

    # 反射补搜 → 执行检索（直接搜新查询，跳过plan_retrieval）
    workflow.add_edge("reflect_search", "execute_search")

    # 生成 → 结束
    workflow.add_edge("generate_answer", END)

    logger.info("Agentic RAG 工作流构建完成（7节点 + 2条件边）")
    return workflow.compile(checkpointer=checkpointer)


async def run_agent(
    query: str,
    file_paths=None,
    config=None,
    session_id: str = "default",
    checkpointer: Optional[MemorySaver] = None,
    history: Optional[list] = None,
    long_term_memories: Optional[list] = None,
    max_iterations: Optional[int] = None,
):
    """便捷调用入口（含记忆支持）

    Args:
        query: 用户问题
        file_paths: 关联文件路径列表
        config: AgenticRAGConfig 实例
        session_id: 会话 ID
        checkpointer: LangGraph Checkpointer
        history: 对话历史
        long_term_memories: 检索到的长期记忆
        max_iterations: 最大迭代次数（优先级高于 config）

    Returns:
        AgenticRAGState 最终状态
    """
    app = build_agent(checkpointer)
    initial = create_initial_state(
        query=query,
        file_paths=file_paths,
        session_id=session_id,
        history=history or [],
        long_term_memories=long_term_memories or [],
    )
    if max_iterations is not None:
        initial["max_iterations"] = max_iterations
    elif config:
        initial["max_iterations"] = getattr(config, "max_iterations", 2)

    run_kwargs = {"input": initial}
    if checkpointer:
        run_kwargs["config"] = {"configurable": {"thread_id": session_id}}

    result = await app.ainvoke(**run_kwargs)
    return result
