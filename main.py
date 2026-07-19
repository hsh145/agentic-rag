"""
Agentic RAG — FastAPI 服务入口（生产版）

启动：
    cd agentic-rag
    python -m uvicorn main:app --reload --port 8000

API 文档：
    http://localhost:8000/docs
"""
import os
import time
import uuid as uuid_mod
from pathlib import Path
from contextlib import asynccontextmanager

# ==== 第一优先级：从 .env 文件读取 API Key ====
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("DASHSCOPE_API_KEY="):
                _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                if _val:
                    os.environ["DASHSCOPE_API_KEY"] = _val
            elif _line.startswith("MOONSHOT_API_KEY="):
                _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                if _val:
                    os.environ["MOONSHOT_API_KEY"] = _val

# ==== loguru 日志系统（替换标准 logging）====
from loguru import logger
import sys

# 移除默认 handler
logger.remove()
# 控制台输出（彩色）
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan> | {message}",
    level="INFO",
    colorize=True,
)
# 文件输出（轮转）
logger.add(
    "logs/rag_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name} | {message}",
    level="DEBUG",
    rotation="500 MB",
    retention=7,
    compression="gz",
    encoding="utf-8",
)


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json

from agent.graph import build_agent, run_agent
from agent.state import create_initial_state
from agent.memory import AgentMemory
from config import DEFAULT_CONFIG

# ============================================================
# 全局状态
# ============================================================
agent_memory = AgentMemory(db_path="./data/memory.db", max_turns=20)


class ServiceStatus:
    """服务组件健康状态"""
    index_loaded: bool = False
    embedding_ready: bool = False
    memory_ready: bool = True


service_status = ServiceStatus()
logger.info("AgentMemory 全局实例已创建")


# ============================================================
# Pydantic 模型
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    file_paths: list[str] = Field(default=[], description="文件路径列表（可选）")
    max_iterations: int = Field(default=2, ge=1, le=5, description="最大检索迭代次数")
    session_id: str = Field(default="", description="会话 ID（留空自动生成）")


class QueryResponse(BaseModel):
    success: bool
    answer: str = ""
    sources: list[str] = []
    iterations: int = 0
    chunk_count: int = 0
    session_id: str = ""
    memory_stats: dict = {}
    elapsed_ms: float = 0
    error: str = ""


class TraceResponse(QueryResponse):
    """支持溯源追踪的响应 — 包含逐跳轨迹和完整 chunk 数据"""
    agentic_trace: list = Field(default_factory=list)
    retrieved_chunks: list = Field(default_factory=list)
    evidence_scores: dict = Field(default_factory=dict)
    evidence_feedback: str = ""
    missing_gaps: list = Field(default_factory=list)
    retrieval_plan: list = Field(default_factory=list)
    supplementary_queries: list = Field(default_factory=list)


# ============================================================
# FastAPI 应用
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MOONSHOT_API_KEY"))
    if not has_key:
        logger.error("未配置 API Key！请在 .env 文件中设置 DASHSCOPE_API_KEY")
    else:
        logger.info("API Key 已配置")

    try:
        from rag.embedder import EmbeddingManager
        embedder = EmbeddingManager(DEFAULT_CONFIG.embedding_model)
        agent_memory.set_embedding_model(embedder.get_embeddings())
        service_status.embedding_ready = True
        logger.info("长期记忆 Embedding 模型已注入")
    except Exception as e:
        logger.warning(f"长期记忆 Embedding 注入跳过: {e}")

    logger.info("Agentic RAG 服务启动")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="Agentic RAG API",
    description="多格式 Agentic RAG 检索系统 — 支持 PDF/Word/Excel/图片/文本文件的自动解析与智能检索",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 中间件：Trace ID + 请求计时
# ============================================================
@app.middleware("http")
async def add_trace_id_middleware(request: Request, call_next):
    trace_id = f"req_{uuid_mod.uuid4().hex[:12]}"
    request.state.trace_id = trace_id
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    logger.info(f"[{trace_id}] {request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.0f}ms)")
    response.headers["X-Trace-ID"] = trace_id
    return response


# ============================================================
# 全局异常处理
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.error(f"[{trace_id}] 未捕获异常: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={
        "success": False, "error": f"服务器内部错误: {type(exc).__name__}", "trace_id": trace_id,
    })


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.warning(f"[{trace_id}] 参数错误: {exc}")
    return JSONResponse(status_code=400, content={
        "success": False, "error": str(exc), "trace_id": trace_id,
    })


@app.get("/")
def root():
    return {
        "service": "Agentic RAG",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "POST /api/ask": "提问（含记忆支持）",
            "POST /api/ask/trace": "溯源问答（含完整分块可视化数据）",
            "GET  /api/health": "健康检查",
            "GET  /api/memory/stats": "记忆统计",
            "DELETE /api/memory/{session_id}": "清除指定会话记忆",
            "POST /api/feedback": "提交反馈（好评/差评/错误类型）",
            "GET  /api/feedback": "查询历史反馈",
        },
    }


@app.get("/api/health")
def health():
    """增强的健康检查"""
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MOONSHOT_API_KEY"))
    mem_stats = agent_memory.stats()

    # 检查 FAISS 索引状态
    index_check = {"loaded": service_status.index_loaded}
    try:
        from rag.embedder import EmbeddingManager
        from rag.indexer import IndexManager
        from config import DEFAULT_CONFIG as cfg
        embedder = EmbeddingManager(cfg.embedding_model)
        idx_mgr = IndexManager(embedder.get_embeddings(), cfg.index_save_path)
        index_check["loaded"] = idx_mgr.load_index()
        if index_check["loaded"]:
            index_check["total_vectors"] = idx_mgr.index.ntotal
    except Exception:
        index_check["loaded"] = False

    service_status.index_loaded = index_check["loaded"]

    return {
        "status": "ok",
        "service": "agentic-rag",
        "version": "1.1.0",
        "api_key_configured": has_key,
        "checks": {
            "embedding": service_status.embedding_ready,
            "memory": service_status.memory_ready,
            "index": index_check,
        },
        "memory": {
            "session_count": mem_stats["session_count"],
            "fact_count": mem_stats["fact_count"],
        },
    }


@app.get("/api/memory/stats")
def memory_stats():
    """返回记忆系统统计信息"""
    stats = agent_memory.stats()
    return {"success": True, "data": stats}


@app.delete("/api/memory/{session_id}")
def clear_memory(session_id: str):
    """清除指定会话的记忆"""
    agent_memory.clear_session(session_id)
    return {"success": True, "message": f"会话 {session_id} 的记忆已清除"}


class ForgetRequest(BaseModel):
    strategy: str = "old"  # old | low_confidence | duplicates
    days: int = 30
    threshold: float = 0.3


@app.post("/api/memory/forget")
def forget_memory(req: ForgetRequest):
    """遗忘机制：按策略清理长期记忆"""
    try:
        if req.strategy == "old":
            count = agent_memory.forget_old_facts(req.days)
        elif req.strategy == "low_confidence":
            count = agent_memory.forget_low_confidence(req.threshold)
        elif req.strategy == "duplicates":
            count = agent_memory.forget_duplicates()
        else:
            return {"success": False, "error": f"未知策略: {req.strategy}"}
        return {"success": True, "deleted": count, "strategy": req.strategy}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 反馈收集（文件存储，用于产品闭环）
# ============================================================
_FEEDBACK_FILE = Path(__file__).parent / "data" / "feedback.jsonl"


class FeedbackEntry(BaseModel):
    session_id: str = ""
    question: str = ""
    answer_snippet: str = ""
    rating: int = Field(default=0, ge=-1, le=1)  # -1 差评, 0 未评, 1 好评
    error_type: str = ""  # hallucination|missing_info|wrong|other|""
    notes: str = ""
    source: str = ""  # 来自哪个页面/功能


@app.post("/api/feedback")
async def submit_feedback(fb: FeedbackEntry):
    """提交反馈"""
    try:
        _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = fb.model_dump()
        record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(_FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"反馈已记录: rating={fb.rating}")
        return {"success": True}
    except Exception as e:
        logger.warning(f"反馈记录失败: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/feedback")
async def get_feedback(limit: int = 50):
    """查询历史反馈"""
    if not _FEEDBACK_FILE.exists():
        return {"success": True, "data": []}
    records = []
    with open(_FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.reverse()
    return {"success": True, "data": records[:limit]}


@app.post("/api/ask", response_model=QueryResponse)
async def ask(request: QueryRequest, http_request: Request = None):
    """处理用户问题（含记忆支持）"""
    # 生成或使用 session_id
    session_id = request.session_id.strip() or f"session_{uuid_mod.uuid4().hex[:12]}"

    t_start = time.time()
    try:
        logger.info(f"[{session_id}] 收到请求: query={request.query[:50]}...")

        # ==== 1. 加载记忆 ====
        history = agent_memory.load_history(session_id, limit=6)
        long_term_memories = agent_memory.recall_facts(request.query, top_k=3)

        if history:
            logger.info(f"[{session_id}] 加载了 {len(history)} 条历史记录")
        if long_term_memories:
            logger.info(f"[{session_id}] 召回 {len(long_term_memories)} 条长期记忆")

        # ==== 2. 注入 LLM 用于事实提取 ====
        from langchain_community.chat_models import ChatTongyi
        _base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
        _llm_kwargs = dict(
            model="qwen-turbo", temperature=0.1,
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        )
        if _base_url:
            _llm_kwargs["dashscope_api_base"] = _base_url
        fact_llm = ChatTongyi(**_llm_kwargs)

        # ==== 3. 运行 Agent ====
        result = await run_agent(
            query=request.query,
            file_paths=request.file_paths or [],
            config=DEFAULT_CONFIG,
            session_id=session_id,
            checkpointer=agent_memory.get_checkpointer(),
            history=history,
            long_term_memories=long_term_memories,
            max_iterations=request.max_iterations,
        )

        if result.get("error"):
            elapsed = (time.time() - t_start) * 1000
            return QueryResponse(
                success=False,
                error=result["error"],
                session_id=session_id,
                elapsed_ms=round(elapsed, 1),
            )

        answer = result.get("final_answer", "")

        # ==== 4. 保存短期记忆 ====
        agent_memory.save_turn(
            session_id=session_id,
            query=request.query,
            answer=answer,
            sources=result.get("sources", []),
        )

        # ==== 5. 提取并存储长期记忆 ====
        stored = agent_memory.extract_and_store(
            session_id=session_id,
            query=request.query,
            answer=answer,
            llm=fact_llm,
        )

        elapsed = (time.time() - t_start) * 1000

        return QueryResponse(
            success=True,
            answer=answer,
            sources=result.get("sources", []),
            iterations=result.get("iteration", 0),
            chunk_count=len(result.get("retrieved_chunks", [])),
            session_id=session_id,
            memory_stats={
                "history_len": len(history) // 2,
                "facts_recalled": len(long_term_memories),
                "facts_stored": stored,
            },
            elapsed_ms=round(elapsed, 1),
        )

    except Exception as e:
        elapsed = (time.time() - t_start) * 1000
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[{session_id}] 处理请求出错: {e}\n{tb}")
        return QueryResponse(success=False, error=str(e), session_id=session_id, elapsed_ms=round(elapsed, 1))


@app.post("/api/ask/trace", response_model=TraceResponse)
async def ask_trace(request: QueryRequest, http_request: Request = None):
    """处理用户问题并返回完整溯源追踪数据（含逐跳轨迹 + chunk 详情）

    与 /api/ask 的区别：
    - 返回完整的 agentic_trace（包含意图分析→检索→评估→补搜→生成的每步记录）
    - 返回所有 retrieved_chunks 及其分数、来源
    - 返回 evidence_scores / missing_gaps 等决策信息
    适合溯源问答、分块可视化观测场景。
    """
    session_id = request.session_id.strip() or f"trace_{uuid_mod.uuid4().hex[:12]}"
    t_start = time.time()

    try:
        logger.info(f"[TRACE:{session_id}] query={request.query[:50]}...")

        # ==== 记忆加载 ====
        history = agent_memory.load_history(session_id, limit=6)
        long_term_memories = agent_memory.recall_facts(request.query, top_k=3)

        from langchain_community.chat_models import ChatTongyi
        _base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
        _llm_kwargs = dict(
            model="qwen-turbo", temperature=0.1,
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        )
        if _base_url:
            _llm_kwargs["dashscope_api_base"] = _base_url
        fact_llm = ChatTongyi(**_llm_kwargs)

        # ==== 运行 Agent ====
        result = await run_agent(
            query=request.query,
            file_paths=request.file_paths or [],
            config=DEFAULT_CONFIG,
            session_id=session_id,
            checkpointer=agent_memory.get_checkpointer(),
            history=history,
            long_term_memories=long_term_memories,
            max_iterations=request.max_iterations,
        )

        if result.get("error"):
            elapsed = (time.time() - t_start) * 1000
            return TraceResponse(
                success=False,
                error=result["error"],
                session_id=session_id,
                elapsed_ms=round(elapsed, 1),
            )

        answer = result.get("final_answer", "")

        # ==== 保存记忆 ====
        agent_memory.save_turn(
            session_id=session_id,
            query=request.query,
            answer=answer,
            sources=result.get("sources", []),
        )
        stored = agent_memory.extract_and_store(
            session_id=session_id,
            query=request.query,
            answer=answer,
            llm=fact_llm,
        )

        elapsed = (time.time() - t_start) * 1000

        # 构建用于可视化的结构化 trace
        trace = result.get("agentic_trace", [])

        # 给每条 trace 补上 readable 的 stage_name
        stage_names = {
            "parse_intent": "意图分析",
            "plan_retrieval": "检索规划",
            "execute_search": "执行检索",
            "evaluate_evidence": "证据评估",
            "reflect_search": "反射补搜",
            "generate_answer": "生成回答",
        }
        for entry in trace:
            entry["stage_label"] = stage_names.get(entry.get("type", ""), entry.get("type", ""))

        # 精简 chunk 数据（只保留可视化所需字段）
        chunks_trimmed = []
        for c in result.get("retrieved_chunks", []):
            chunks_trimmed.append({
                "content_snippet": c.get("content", "")[:300],
                "score": c.get("score", 0),
                "source": c.get("metadata", {}).get("source", "unknown"),
                "source_type": c.get("metadata", {}).get("source_type", "text"),
            })

        return TraceResponse(
            success=True,
            answer=answer,
            sources=result.get("sources", []),
            iterations=result.get("iteration", 0),
            chunk_count=len(result.get("retrieved_chunks", [])),
            session_id=session_id,
            memory_stats={
                "history_len": len(history) // 2,
                "facts_recalled": len(long_term_memories),
                "facts_stored": stored,
            },
            elapsed_ms=round(elapsed, 1),
            # === 溯源追踪数据 ===
            agentic_trace=trace,
            retrieved_chunks=chunks_trimmed,
            evidence_scores=result.get("evidence_scores", {}),
            evidence_feedback=result.get("evidence_feedback", ""),
            missing_gaps=result.get("missing_gaps", []),
            retrieval_plan=result.get("retrieval_plan", []),
            supplementary_queries=result.get("supplementary_queries", []),
        )

    except Exception as e:
        elapsed = (time.time() - t_start) * 1000
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[TRACE:{session_id}] 出错: {e}\n{tb}")
        return TraceResponse(success=False, error=str(e), session_id=session_id, elapsed_ms=round(elapsed, 1))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
