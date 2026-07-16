@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ============================================
echo  Agentic RAG — 启动 Streamlit 前端
echo ============================================
echo.
echo 确保后端已启动：uvicorn main:app --reload --port 8000
echo.
streamlit run web/app.py --server.port 8501 --server.headless true
pause
