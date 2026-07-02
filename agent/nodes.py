"""
LangGraph 节点定义 - 6 个节点实现 Agentic RAG 全流程
"""
import json
import os
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi

from .state import AgenticRAGState
from .tools import DocumentParserTool, RAGRetrievalTool, EvidenceEvaluator

logger = logging.getLogger("agent.nodes")


def _get_llm():
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        key = os.environ.get("MOONSHOT_API_KEY", "")
    return ChatTongyi(
        model="qwen-turbo", temperature=0.1,
        dashscope_api_key=key,
    )


def _get_llm_for_generate():
    """生成回答专用的 LLM（使用 qwen-max，质量更高）"""
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    return ChatTongyi(
        model="qwen-max", temperature=0.3,
        dashscope_api_key=key,
    )


def get_api_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("MOONSHOT_API_KEY", "")


# ============================================================
# 节点1：意图分析 (parse_intent)
# ============================================================
def parse_intent(state: AgenticRAGState) -> Dict[str, Any]:
    """分析用户意图：是简单问答、文件处理还是深度检索"""
    query = state["query"]
    file_paths = state.get("file_paths", [])

    llm = _get_llm()
    prompt = f"""分析用户问题，输出 JSON：

用户问题：{query}
用户提供的文件：{file_paths if file_paths else "无"}

请判断：
1. 是否需要解析文件（有文件路径、或问题中提到了具体文件）
2. 是否需要检索知识库
3. 是否需要联网搜索
4. 需求类型：simple（简单问答）/ file_process（文件处理）/ deep_search（深度检索）

输出JSON：
{{
    "need_file_parse": true/false,
    "need_rag_search": true/false,
    "need_web_search": true/false,
    "query_type": "simple|file_process|deep_search",
    "parsed_file_paths": ["从问题中提取的文件路径"],
    "analysis": "简要分析"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "need_file_parse": False,
            "need_rag_search": True,
            "need_web_search": False,
            "query_type": "deep_search",
            "parsed_file_paths": [],
            "analysis": "无法解析，走默认检索流程",
        }

    return {
        "parsed_documents": [],
        "messages": [
            HumanMessage(content=query),
            AIMessage(content=f"[意图分析] {parsed.get('analysis', '')}"),
        ],
    }


# ============================================================
# 节点2：文件解析 (parse_files)
# ============================================================
def parse_files(state: AgenticRAGState) -> Dict[str, Any]:
    """解析用户提供的文件"""
    query = state["query"]
    file_paths = state.get("file_paths", [])

    # 从 query 中用正则提取文件路径
    import re
    path_pattern = r'[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*|[A-Za-z]:/(?:[^/\s]+/)*[^/\s]*'
    found_paths = re.findall(path_pattern, query)
    all_paths = list(set(file_paths + found_paths))

    if not all_paths:
        return {"parsed_documents": [], "parsing_errors": ["未找到文件路径"]}

    parser_tool = DocumentParserTool(api_key=get_api_key())
    docs, errors = parser_tool.parse_files(all_paths)

    return {
        "parsed_documents": [{"content": d.page_content, "metadata": d.metadata} for d in docs],
        "parsing_errors": errors,
    }


# ============================================================
# 节点3：检索规划 (plan_retrieval)
# ============================================================
def plan_retrieval(state: AgenticRAGState) -> Dict[str, Any]:
    """制定检索计划：将复杂查询拆解为多个子查询"""
    query = state["query"]
    docs = state.get("parsed_documents", [])

    llm = _get_llm()
    context_hint = f"（已解析 {len(docs)} 个文档）" if docs else ""

    prompt = f"""你是检索规划专家。将以下用户问题拆解为 1-3 个独立的搜索子查询。
每个子查询应该覆盖问题的一个独立维度。

用户问题: {query}
{context_hint}

输出 JSON（只输出 JSON）：
{{
    "sub_queries": ["子查询1", "子查询2"],
    "reasoning": "拆解思路"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
        sub_queries = parsed.get("sub_queries", [query])
    except json.JSONDecodeError:
        sub_queries = [query]

    return {
        "retrieval_plan": sub_queries if sub_queries else [query],
        "messages": [AIMessage(content=f"[检索规划] {len(sub_queries)} 个子查询")],
    }


# ============================================================
# 节点4：执行检索 (execute_search)
# ============================================================
def execute_search(state: AgenticRAGState) -> Dict[str, Any]:
    """执行多路检索"""
    from rag.embedder import EmbeddingManager
    from rag.indexer import IndexManager
    from rag.retriever import HybridRetriever
    from rag import get_chunker
    from langchain_core.documents import Document as LCDocument
    from config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG
    query = state["query"]
    plan = state.get("retrieval_plan", [query])
    docs_data = state.get("parsed_documents", [])

    embedder = EmbeddingManager(config.embedding_model)
    indexer = IndexManager(embedder.get_embeddings(), config.index_save_path)

    # 加载已有索引
    has_existing = indexer.load_index()

    # 如果有解析的新文档，追加到索引
    if docs_data:
        raw_docs = [
            LCDocument(page_content=d["content"], metadata=d["metadata"])
            for d in docs_data
        ]
        chunker_cls = get_chunker()
        chunker = chunker_cls()
        new_chunks = chunker.chunk_all(raw_docs)

        if has_existing:
            indexer.add_documents(new_chunks)
        else:
            indexer.build_index(new_chunks)
        indexer.save_index()
    elif not has_existing:
        return {"retrieved_chunks": [], "messages": [AIMessage(content="[检索] 无可用索引，请先添加文档")]}

    # 构建混合检索器
    all_chunks = getattr(indexer, "documents", [])
    retriever = HybridRetriever(indexer, all_chunks, rrf_k=config.rrf_k)

    # 执行所有子查询
    all_results = []
    seen = set()
    for sub_q in plan:
        results = retriever.hybrid_search(sub_q, top_k=config.top_k)
        for doc in results:
            key = hash(doc.page_content[:200])
            if key not in seen:
                seen.add(key)
                all_results.append({
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": doc.metadata.get("rrf_score", 0),
                })

    return {
        "retrieved_chunks": all_results,
        "messages": [AIMessage(content=f"[检索] 共 {len(all_results)} 个结果")],
    }


# ============================================================
# 节点5：证据评估 (evaluate_evidence)
# ============================================================
def evaluate_evidence(state: AgenticRAGState) -> Dict[str, Any]:
    """评估检索结果的充分性"""
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    scores, feedback, needs_more = EvidenceEvaluator.evaluate(
        [chunk for chunk in []], query
    )

    # 用实际数据评估
    total_chars = sum(len(c.get("content", "")) for c in chunks)
    needs_more = (len(chunks) < 2 and total_chars < 500) and (iteration < max_iter)

    return {
        "evidence_scores": {
            "chunk_count": len(chunks),
            "total_chars": total_chars,
        },
        "evidence_feedback": f"检索到 {len(chunks)} 块，共 {total_chars} 字",
        "needs_more": needs_more,
        "iteration": iteration + 1,
    }


# ============================================================
# 节点6：生成回答 (generate_answer)
# ============================================================
def generate_answer(state: AgenticRAGState) -> Dict[str, Any]:
    """多模式生成最终回答"""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    docs_data = state.get("parsed_documents", [])
    feedback = state.get("evidence_feedback", "")

    # 构建上下文
    context_parts = []
    sources = []

    for chunk in chunks:
        content = chunk.get("content", "")
        meta = chunk.get("metadata", {})
        source = meta.get("source", "") or meta.get("file_name", "未知来源")
        src_type = meta.get("source_type", "text")
        page = meta.get("page", "")

        if src_type == "table":
            context_parts.append(f"[表格] {content}")
        elif src_type == "image_description":
            context_parts.append(f"[图片] {content}")
        else:
            context_parts.append(content)

        source_label = f"{Path(source).name}" if source else "未知"
        if page:
            source_label += f" (第{page}页)"
        sources.append(source_label)

    # 文件内容也加到上下文
    for d in docs_data:
        context_parts.append(d.get("content", ""))

    context = "\n\n---\n\n".join(context_parts) if context_parts else "（无检索结果）"
    sources = list(set(sources))

    llm = _get_llm_for_generate()
    prompt = f"""你是一个智能问答助手。请根据以下检索到的信息回答用户问题。

用户问题：{query}

参考信息：
{context[:12000]}

检索评估：{feedback}

要求：
- 基于参考信息回答，不要编造
- 如果参考信息不够，明确告知哪些方面信息不足
- 如果包含表格数据，用表格形式呈现
- 如果包含图片描述，在回答中提及
- 在回答末尾列出参考来源"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_answer": response.content,
        "sources": sources,
        "completed": True,
        "messages": [AIMessage(content=response.content)],
    }


# ============================================================
# 路由函数
# ============================================================
def route_after_intent(state: AgenticRAGState) -> str:
    """根据意图分析结果路由"""
    file_paths = state.get("file_paths", [])
    # 如果有文件路径或者 parsed_documents 有内容（说明 parse_intent 阶段检测到了文件）
    if file_paths:
        return "parse_files"
    return "plan_retrieval"


def route_after_search(state: AgenticRAGState) -> str:
    """评估后决定是否补搜"""
    if state.get("needs_more", False):
        return "plan_retrieval"
    return "generate_answer"


# 用于 Windows 路径处理
from pathlib import Path
