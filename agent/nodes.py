"""
LangGraph 节点定义 - 6 个节点实现 Agentic RAG 全流程
"""
import json
import os
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi

from .state import AgenticRAGState
from .tools import DocumentParserTool
from config import DEFAULT_CONFIG

logger = logging.getLogger("agent.nodes")

# ---- 本地模型缓存（全局单例） ----
_local_judge_llm = None


def _get_local_judge():
    """获取本地判断模型（延迟初始化，第一次调用时加载）"""
    global _local_judge_llm
    if _local_judge_llm is None and DEFAULT_CONFIG.use_local_judge:
        from models.local_llm import LocalLLM
        _local_judge_llm = LocalLLM.from_config(DEFAULT_CONFIG)
    return _local_judge_llm


def _get_llm():
    """获取 LLM — 优先使用本地模型"""
    # 如果启用本地模型且有缓存，返回 None 占位，调用处会走 local judge
    local = _get_local_judge()
    if local is not None:
        # 返回 None 表示使用本地模型（调用处判断）
        return None

    # 回退到 API 模型
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
    if not key:
        key = os.environ.get("MOONSHOT_API_KEY", "")
    kwargs = dict(model="qwen-turbo", temperature=0.1, dashscope_api_key=key)
    if base_url:
        kwargs["dashscope_api_base"] = base_url
    return ChatTongyi(**kwargs)


def _call_llm_json(prompt: str) -> dict:
    """通用 JSON 输出调用 — 自动选择本地/API 模型

    优先用本地 judge 模型，失败或未配置时回退到 API。
    """
    local = _get_local_judge()
    if local is not None:
        try:
            result = local.generate_json(prompt)
            if result:
                return result
        except Exception as e:
            logger.warning(f"本地模型调用失败，回退 API: {e}")

    # 回退：API 调用
    from langchain_core.messages import HumanMessage
    llm = ChatTongyi(
        model="qwen-turbo", temperature=0.1,
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _get_llm_for_generate():
    """生成回答专用的 LLM（使用 qwen-max，质量更高）"""
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
    kwargs = dict(model="qwen-max", temperature=0.3, dashscope_api_key=key)
    if base_url:
        kwargs["dashscope_api_base"] = base_url
    return ChatTongyi(**kwargs)


def get_api_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("MOONSHOT_API_KEY", "")


def _rewrite_query(query: str, history: List[Dict], llm=None) -> str:
    """指代消解 + query 补全：将带指代/省略的 query 改写为独立可检索的 query

    例如：
      history=["SFT 的 learning rate 是多少？", "推荐 2e-5"]
      query="那 batch size 呢？"
      → "SFT 微调中 batch size 的推荐值是多少？"
    """
    if not history:
        return query

    history_text = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:200]}"
        for m in history[-6:]
    )

    prompt = f"""你是对话指代消解专家。将用户的最新问题改写为可独立检索的查询。

对话历史：
{history_text}

用户最新问题：{query}

要求：
- 如果问题包含指代（它、这个、那、其等）或省略，补全为完整查询
- 如果问题可以独立理解，保持原样
- 输出 JSON：{{"rewritten_query": "改写后的查询"}}"""

    parsed = _call_llm_json(prompt)
    rewritten = parsed.get("rewritten_query", "") if parsed else ""
    if rewritten and rewritten != query:
        logger.info(f"Query Rewrite: '{query[:50]}' → '{rewritten[:50]}'")
        return rewritten
    return query


# ============================================================
# 节点1：意图分析 (parse_intent)
# ============================================================
def parse_intent(state: AgenticRAGState) -> Dict[str, Any]:
    """分析用户意图：是简单问答、文件处理还是深度检索"""
    query = state["query"]
    file_paths = state.get("file_paths", [])

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

    parsed = _call_llm_json(prompt)
    if not parsed:
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
        "agentic_trace": [{
            "hop": 0,
            "type": "parse_intent",
            "need_file_parse": parsed.get("need_file_parse", False),
            "need_rag_search": parsed.get("need_rag_search", True),
            "query_type": parsed.get("query_type", "deep_search"),
            "analysis": parsed.get("analysis", ""),
        }],
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
    history = state.get("history", [])
    docs = state.get("parsed_documents", [])

    # ---- 第一轮：如果有历史，先做 query rewrite（指代消解）----
    rewritten = _rewrite_query(query, history)
    if rewritten and rewritten != query:
        query = rewritten

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

    parsed = _call_llm_json(prompt)
    sub_queries = parsed.get("sub_queries", [query]) if parsed else [query]

    return {
        "query": query,  # 覆盖为改写后的 query，后续节点使用
        "retrieval_plan": sub_queries if sub_queries else [query],
        "agentic_trace": [{
            "hop": 0,
            "type": "plan_retrieval",
            "sub_queries": sub_queries if sub_queries else [query],
            "reasoning": parsed.get("reasoning", "") if parsed else "",
            "was_rewritten": rewritten != state["query"] if rewritten else False,
        }],
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
    from rag.reranker import Reranker
    from rag import get_chunker
    from langchain_core.documents import Document as LCDocument
    from config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG
    query = state["query"]
    # 合并常规检索计划和反射补搜的新查询
    plan = list(dict.fromkeys(state.get("retrieval_plan", [query]) + state.get("supplementary_queries", [])))
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

    # 构建混合检索器（可选 reranker）
    all_chunks = getattr(indexer, "documents", [])
    reranker = Reranker(
        model_name=config.rerank_model,
        enabled=config.enable_rerank,
        top_k=config.rerank_top_k,
        device=config.rerank_device,
    )
    retriever = HybridRetriever(indexer, all_chunks, rrf_k=config.rrf_k, reranker=reranker)

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
        "executed_queries": list(dict.fromkeys(state.get("executed_queries", []) + plan)),
        "agentic_trace": [{
            "hop": state.get("iteration", 0) + 1,
            "type": "execute_search",
            "queries": plan,
            "chunk_count": len(all_results),
            "top_chunks": [
                {
                    "content_snippet": r["content"][:200],
                    "score": r["score"],
                    "source": r["metadata"].get("source", "unknown"),
                }
                for r in all_results[:5]  # 前5个详情的快照
            ],
        }],
        "messages": [AIMessage(content=f"[检索] 共 {len(all_results)} 个结果")],
    }


# ============================================================
# 节点5：证据评估 (evaluate_evidence) — LLM驱动
# ============================================================
def evaluate_evidence(state: AgenticRAGState) -> Dict[str, Any]:
    """LLM 评估检索结果是否充分，识别具体信息缺口

    这是 Agentic RAG 与普通 RAG 的核心区别点：
    - 普通 RAG: 搜一次就答，不管信息够不够
    - Agentic:  搜→LLM评估→不够→补搜→再评估→够了→生成
    """
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if not chunks:
        return {
            "evidence_scores": {"chunk_count": 0, "total_chars": 0},
            "evidence_feedback": "未检索到任何内容",
            "needs_more": iteration < max_iter,
            "missing_gaps": ["需要重新检索"],
            "iteration": iteration + 1,
        }

    # 构建评估上下文
    context_parts = []
    for i, c in enumerate(chunks):
        content = c.get("content", "")[:500]
        source = c.get("metadata", {}).get("source", f"chunk_{i}")
        context_parts.append(f"[来源 {i+1}] {source}\n{content}")

    context = "\n\n".join(context_parts)
    total_chars = sum(len(c.get("content", "")) for c in chunks)
    chunk_count = len(chunks)

    prompt = f"""你是证据评估专家。判断当前检索到的信息是否足够回答用户问题。

用户问题：{query}

检索到的 {chunk_count} 条证据：
{context[:6000]}

请分析并输出 JSON（只输出 JSON）：
{{
    "can_answer": true/false,        // 现有信息能否完整回答问题
    "missing_gaps": ["缺口1", "缺口2"], // 如果 can_answer=false，列出具体缺什么信息
    "issue_type": "",                 // 如果 can_answer=false，判断原因：
                                      //   "bad_query" — query 有指代/省略，需结合历史改写后再搜
                                      //   "multi_hop" — 需要先找到某个实体再基于它跳转搜索
                                      //   "insufficient" — 单纯信息量不够，需更多同类内容
                                      //   如果 can_answer=true，留空
    "feedback": "评估摘要（100字内）",    // 简要说明信息充分/不足之处
    "confidence": 0.0-1.0             // 对回答质量的置信度
}}"""

    parsed = _call_llm_json(prompt)
    if not parsed:
        # 回退：简单规则判断
        logger.warning("LLM 证据评估失败，回退规则评估")
        parsed = {
            "can_answer": chunk_count >= 2 and total_chars >= 300,
            "missing_gaps": [],
            "issue_type": "insufficient" if chunk_count < 2 else "",
            "feedback": f"检索到 {chunk_count} 块，共 {total_chars} 字（规则评估）",
            "confidence": 0.5,
        }

    needs_more = (not parsed["can_answer"]) and (iteration < max_iter)
    issue_type = parsed.get("issue_type", "")
    if issue_type not in ("bad_query", "multi_hop", "insufficient"):
        issue_type = "insufficient" if needs_more else ""

    return {
        "evidence_scores": {
            "chunk_count": chunk_count,
            "total_chars": total_chars,
            "confidence": parsed.get("confidence", 0.5),
        },
        "evidence_feedback": parsed.get("feedback", ""),
        "needs_more": needs_more,
        "missing_gaps": parsed.get("missing_gaps", []),
        "issue_type": issue_type,
        "iteration": iteration + 1,
        "agentic_trace": [{
            "hop": iteration + 1,
            "type": "evaluate_evidence",
            "chunk_count": chunk_count,
            "total_chars": total_chars,
            "can_answer": parsed.get("can_answer", False),
            "confidence": parsed.get("confidence", 0.5),
            "feedback": parsed.get("feedback", ""),
            "missing_gaps": parsed.get("missing_gaps", []),
            "needs_more": needs_more,
        }],
    }


# ============================================================
# 节点5b：反射补搜 (reflect_search)
# ============================================================
def reflect_search(state: AgenticRAGState) -> Dict[str, Any]:
    """根据信息不足的原因，走三种不同的补搜路径

    bad_query:      query 有指代/省略 → 用历史改写 query，替换原计划
    multi_hop:      需要基于已搜到的实体跳转搜索 → 生成实体感知的新查询
    insufficient:   单纯信息量不够 → 生成针对缺口的补充查询（原逻辑）
    """
    query = state["query"]
    missing_gaps = state.get("missing_gaps", [])
    executed_queries = state.get("executed_queries", [])
    chunks = state.get("retrieved_chunks", [])
    history = state.get("history", [])
    issue_type = state.get("issue_type", "")
    all_executed = list(executed_queries)

    # ================================================================
    # 路径 A：query 有指代/省略 → 用历史改写，替换原计划
    # ================================================================
    if issue_type == "bad_query":
        rewritten = _rewrite_query(query, history)
        if rewritten and rewritten != query and rewritten not in all_executed:
            logger.info(f"[bad_query] 改写 '{query[:40]}' → '{rewritten[:40]}'")
            all_executed.append(rewritten)
            return {
                "query": rewritten,  # 覆盖 query，后续节点用改写后的
                "retrieval_plan": [rewritten],
                "supplementary_queries": [],
                "executed_queries": all_executed,
                "agentic_trace": [{
                    "hop": state.get("iteration", 0),
                    "type": "reflect_search",
                    "issue_type": "bad_query",
                    "original_query": query,
                    "rewritten_query": rewritten,
                    "generated_queries": [rewritten],
                    "supplement_count": 1,
                }],
                "messages": [AIMessage(content=f"[反射补搜] bad_query: 改写为 '{rewritten[:50]}'")],
            }

    # ================================================================
    # 路径 B：多跳查询 → 基于已有 chunk 提取新实体/方向
    # ================================================================
    if issue_type == "multi_hop":
        chunk_snippets = "\n".join(
            f"- {c.get('content','')[:200]}"
            for c in chunks[:5]
        )
        prompt = f"""你是多跳查询专家。已检索到的内容提到了某些实体或线索，
需要基于它们跳转搜索才能找到完整答案。

用户问题：{query}

已检索到的内容：
{chunk_snippets}

信息缺口：
{'、'.join(missing_gaps) if missing_gaps else '无'}

请分析已检索到的内容中提到了哪些需要进一步搜索的实体或方向，
生成 1-2 个针对这些新实体的搜索查询。
要求与已执行过的查询不同：{all_executed}

输出 JSON：{{"supplementary_queries": ["查询1", "查询2"]}}"""
        parsed = _call_llm_json(prompt)
        new_queries = parsed.get("supplementary_queries", []) if parsed else []

    # ================================================================
    # 路径 C：单纯信息量不够（或 issue_type 为空/未知）
    # ================================================================
    else:
        if not missing_gaps:
            # 没有具体缺口，从已有 chunk 反推
            chunk_topics = []
            for c in chunks[:3]:
                meta = c.get("metadata", {})
                src = meta.get("source", "unknown")
                chunk_topics.append(src)

            prompt = f"""你是检索规划专家。以下信息不足以回答用户问题，请生成新的搜索查询。

用户问题：{query}
已检索的来源：{', '.join(chunk_topics)}
已执行过查询：{all_executed}

请生成 1-2 个补充搜索查询，要求：
1. 与已执行查询不同
2. 针对原问题未被覆盖的方面

输出 JSON：{{"supplementary_queries": ["查询1", "查询2"]}}"""
        else:
            gaps_text = "\n".join(f"- {g}" for g in missing_gaps)
            prompt = f"""你是检索规划专家。用户问题缺少关键信息，请生成针对性的搜索查询来填补缺口。

用户问题：{query}

信息缺口：
{gaps_text}

已执行过的查询（避免重复）：
{chr(10).join(f'- {q}' for q in all_executed) if all_executed else '无'}

请为每个缺口生成一个具体的搜索查询。要求：
1. 查询必须直接针对缺口内容
2. 与已执行查询不同
3. 每个查询不要太长，40字以内

输出 JSON：{{"supplementary_queries": ["查询1", "查询2"]}}"""

        parsed = _call_llm_json(prompt)
        new_queries = parsed.get("supplementary_queries", []) if parsed else []

    # 去重
    filtered = [q for q in new_queries if q not in all_executed]

    if filtered:
        logger.info(f"{'[multi_hop]' if issue_type=='multi_hop' else '[insufficient]'} {len(filtered)} 个新查询: {filtered}")
        all_executed.extend(filtered)

    return {
        "supplementary_queries": filtered if filtered else [],
        "executed_queries": all_executed,
        "retrieval_plan": filtered if filtered else state.get("retrieval_plan", [query]),
        "agentic_trace": [{
            "hop": state.get("iteration", 0),
            "type": "reflect_search",
            "issue_type": issue_type or "insufficient",
            "missing_gaps": missing_gaps,
            "generated_queries": filtered,
            "supplement_count": len(filtered),
        }],
        "messages": [AIMessage(content=f"[反射补搜] ({issue_type or 'insufficient'}) 生成 {len(filtered)} 个新查询")],
    }


# ============================================================
# 节点6：生成回答 (generate_answer)
# ============================================================
def generate_answer(state: AgenticRAGState) -> Dict[str, Any]:
    """多模式生成最终回答（含记忆注入）"""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    docs_data = state.get("parsed_documents", [])
    feedback = state.get("evidence_feedback", "")
    long_term_memories = state.get("long_term_memories", [])
    history = state.get("history", [])

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

    # ---- 记忆注入 ----
    memory_section = ""
    if long_term_memories:
        memories_text = "\n".join(f"- {m}" for m in long_term_memories)
        memory_section = f"\n关于用户的历史记忆：\n{memories_text}\n"

    history_section = ""
    if history:
        # 取最近几轮对话历史
        recent = history[-6:] if len(history) > 6 else history
        history_lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:150]
            history_lines.append(f"{role}: {content}")
        history_section = "\n对话历史（最近几轮）：\n" + "\n".join(history_lines) + "\n"

    llm = _get_llm_for_generate()
    prompt = f"""你是一个智能问答助手。请根据以下检索到的信息回答用户问题。

用户问题：{query}
{history_section}
{memory_section}
参考信息：
{context[:10000]}

检索评估：{feedback}

要求：
- 基于参考信息回答，不要编造
- 如果参考信息不够，明确告知哪些方面信息不足
- 如果包含表格数据，用表格形式呈现
- 如果包含图片描述，在回答中提及
- 注意结合对话历史，避免重复之前已完整回答过的内容
- 如果历史记忆与当前参考信息矛盾，以当前参考信息为准
- 在回答末尾列出参考来源"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_answer": response.content,
        "sources": sources,
        "completed": True,
        "agentic_trace": [{
            "hop": state.get("iteration", 0),
            "type": "generate_answer",
            "chunks_used": len(chunks),
            "sources": sources,
            "answer_snippet": response.content[:300],
        }],
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
    """评估后决定是否补搜

    返回:
        "reflect_search" — LLM判断信息不足，走反射补搜
        "generate_answer" — 信息充分，生成回答
    """
    if state.get("needs_more", False):
        return "reflect_search"
    return "generate_answer"


# 用于 Windows 路径处理
from pathlib import Path
