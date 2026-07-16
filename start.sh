#!/bin/bash
# Agentic RAG 启动脚本
# 用法: bash start.sh [backend|frontend|all]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-all}" in
  backend)
    echo "🚀 启动后端服务 (端口 8000)..."
    uvicorn main:app --reload --port 8000
    ;;
  frontend)
    echo "🚀 启动前端服务 (端口 8501)..."
    streamlit run web/app.py --server.port 8501
    ;;
  all)
    echo "============================================"
    echo " Agentic RAG — 一键启动"
    echo "============================================"
    echo ""
    echo "📌 后端: http://localhost:8000"
    echo "📌 前端: http://localhost:8501"
    echo "📌 API 文档: http://localhost:8000/docs"
    echo ""

    # 后台启动后端
    uvicorn main:app --reload --port 8000 &
    BACKEND_PID=$!
    sleep 2

    # 前台启动前端
    streamlit run web/app.py --server.port 8501

    # 退出时清理后端
    kill $BACKEND_PID 2>/dev/null
    ;;
  *)
    echo "用法: bash start.sh [backend|frontend|all]"
    exit 1
    ;;
esac
