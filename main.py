"""
Agentic RAG — FastAPI 服务入口

启动：
    cd agentic-rag
    python -m uvicorn main:app --reload --port 8000

API 文档：
    http://localhost:8000/docs
"""
import os
import logging
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.graph import build_agent
from agent.state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agentic-rag")


# ============================================================
# Pydantic 模型
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    file_paths: list[str] = Field(default=[], description="文件路径列表（可选）")
    max_iterations: int = Field(default=2, ge=1, le=5, description="最大检索迭代次数")


class QueryResponse(BaseModel):
    success: bool
    answer: str = ""
    sources: list[str] = []
    iterations: int = 0
    chunk_count: int = 0
    error: str = ""


# ============================================================
# FastAPI 应用
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MOONSHOT_API_KEY"))
    if not has_key:
        logger.error("❌ 未配置 API Key！请在 .env 文件中设置 DASHSCOPE_API_KEY")
    else:
        logger.info("✅ API Key 已配置")
    logger.info("🚀 Agentic RAG 服务启动")
    yield
    logger.info("👋 服务关闭")


app = FastAPI(
    title="Agentic RAG API",
    description="多格式 Agentic RAG 检索系统 — 支持 PDF/Word/Excel/图片/文本文件的自动解析与智能检索",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Agentic RAG",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /api/ask": "提问（支持文件路径）",
            "GET  /api/health": "健康检查",
        },
    }


@app.get("/api/health")
def health():
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MOONSHOT_API_KEY"))
    return {
        "status": "ok",
        "service": "agentic-rag",
        "api_key_configured": has_key,
    }


@app.post("/api/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """处理用户问题"""
    try:
        logger.info(f"收到请求: query={request.query[:50]}... files={len(request.file_paths)}")

        app_graph = build_agent()

        initial_state = create_initial_state(
            query=request.query,
            file_paths=request.file_paths or [],
        )
        initial_state["max_iterations"] = request.max_iterations

        result = await app_graph.ainvoke(initial_state)

        if result.get("error"):
            return QueryResponse(
                success=False,
                error=result["error"],
            )

        return QueryResponse(
            success=True,
            answer=result.get("final_answer", ""),
            sources=result.get("sources", []),
            iterations=result.get("iteration", 0),
            chunk_count=len(result.get("retrieved_chunks", [])),
        )

    except Exception as e:
        logger.exception("处理请求出错")
        return QueryResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
