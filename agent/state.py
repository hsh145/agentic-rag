"""
Agent 状态定义 — 含 Memory 字段

字段说明：
  短期记忆: session_id + history（当前会话的对话历史）
  长期记忆: long_term_memories（从历史对话中提取并检索到的事实）
"""
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class AgenticRAGState(TypedDict):
    # ===== 用户输入 =====
    query: str                              # 用户问题
    file_paths: List[str]                   # 用户指定的文件路径（可选）
    session_id: str                         # 会话 ID（用于记忆持久化）

    # ===== 记忆 =====
    history: List[Dict[str, Any]]           # 对话历史（来自 SessionMemory）
    long_term_memories: List[str]           # 检索到的长期记忆事实

    # ===== 文件解析结果 =====
    parsed_documents: List[Dict[str, Any]]  # 统一格式的文档列表
    parsing_errors: List[str]               # 解析错误信息

    # ===== 检索计划 =====
    retrieval_plan: List[str]               # 拆解后的子查询列表
    current_sub_query: str                  # 当前正在执行的子查询
    executed_queries: List[str]             # 已执行过的查询（避免重复）
    supplementary_queries: List[str]        # 反射补搜生成的新查询

    # ===== 检索结果 =====
    retrieved_chunks: List[Dict[str, Any]]  # 检索到的文档块

    # ===== 证据评估（LLM驱动） =====
    evidence_scores: Dict[str, float]       # 证据完备性评分
    evidence_feedback: str                  # 评估反馈
    needs_more: bool                        # 是否需要补搜
    missing_gaps: List[str]                 # 具体的信息缺口（LLM识别）
    iteration: int                          # 当前迭代轮次
    max_iterations: int                     # 最大迭代轮次

    # ===== 生成结果 =====
    final_answer: str                       # 最终回答
    sources: List[str]                      # 引用来源列表

    # ===== LLM 消息历史 =====
    messages: Annotated[List[BaseMessage], operator.add]

    # ===== 追踪 =====
    agentic_trace: Annotated[List[Dict[str, Any]], operator.add]  # 逐跳追踪数据（可视化用）

    # ===== 控制 =====
    error: Optional[str]                    # 错误信息
    completed: bool                         # 是否完成


def create_initial_state(
    query: str,
    file_paths: Optional[List[str]] = None,
    session_id: str = "default",
    history: Optional[List[Dict[str, Any]]] = None,
    long_term_memories: Optional[List[str]] = None,
) -> dict:
    return {
        "query": query,
        "file_paths": file_paths or [],
        "session_id": session_id,
        "history": history or [],
        "long_term_memories": long_term_memories or [],
        "parsed_documents": [],
        "parsing_errors": [],
        "retrieval_plan": [],
        "current_sub_query": "",
        "executed_queries": [],
        "supplementary_queries": [],
        "retrieved_chunks": [],
        "evidence_scores": {},
        "evidence_feedback": "",
        "needs_more": False,
        "missing_gaps": [],
        "iteration": 0,
        "max_iterations": 2,
        "final_answer": "",
        "sources": [],
        "agentic_trace": [],
        "messages": [],
        "error": None,
        "completed": False,
    }
