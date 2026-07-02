"""
Agent 状态定义
"""
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class AgenticRAGState(TypedDict):
    # ===== 用户输入 =====
    query: str                              # 用户问题
    file_paths: List[str]                   # 用户指定的文件路径（可选）

    # ===== 文件解析结果 =====
    parsed_documents: List[Dict[str, Any]]  # 统一格式的文档列表
    parsing_errors: List[str]               # 解析错误信息

    # ===== 检索计划 =====
    retrieval_plan: List[str]               # 拆解后的子查询列表
    current_sub_query: str                  # 当前正在执行的子查询

    # ===== 检索结果 =====
    retrieved_chunks: List[Dict[str, Any]]  # 检索到的文档块

    # ===== 证据评估 =====
    evidence_scores: Dict[str, float]       # 证据完备性评分
    evidence_feedback: str                  # 评估反馈（信息缺口）
    needs_more: bool                        # 是否需要补搜
    iteration: int                          # 当前迭代轮次
    max_iterations: int                     # 最大迭代轮次

    # ===== 生成结果 =====
    final_answer: str                       # 最终回答
    sources: List[str]                      # 引用来源列表

    # ===== LLM 消息历史 =====
    messages: Annotated[List[BaseMessage], operator.add]

    # ===== 控制 =====
    error: Optional[str]                    # 错误信息
    completed: bool                         # 是否完成


def create_initial_state(query: str, file_paths: Optional[List[str]] = None) -> dict:
    return {
        "query": query,
        "file_paths": file_paths or [],
        "parsed_documents": [],
        "parsing_errors": [],
        "retrieval_plan": [],
        "current_sub_query": "",
        "retrieved_chunks": [],
        "evidence_scores": {},
        "evidence_feedback": "",
        "needs_more": False,
        "iteration": 0,
        "max_iterations": 2,
        "final_answer": "",
        "sources": [],
        "messages": [],
        "error": None,
        "completed": False,
    }
