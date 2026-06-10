@echo off
chcp 65001 >nul
title UVAIS 海洋牧场智能中枢

echo ==========================================
echo   UVAIS 海洋牧场智能中枢 - 一键启动
echo ==========================================
echo.

REM --- Check .env ---
if not exist .env (
    echo [1/4] Creating .env config...
    copy .env.example .env >nul
    echo   ! Edit .env with your LLM_API_KEY then re-run
    pause
    exit /b 1
)

REM --- Start Milvus ---
echo [1/4] Starting Milvus...
docker-compose up -d
echo   Waiting for Milvus...
timeout /t 5 /nobreak >nul

REM --- Build knowledge base ---
echo [2/4] Checking knowledge base...
python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://localhost:19530'); exit(0 if c.has_collection('marine_local_hybrid_knowledge') else 1)" >nul 2>&1
if errorlevel 1 (
    echo   First run, building knowledge base...
    set PYTHONPATH=.
    python -c "from rag.engine import RagService; RagService().build_knowledge_base()"
) else (
    echo   Knowledge base ready
)

REM --- Start FastAPI ---
echo [3/4] Starting FastAPI server...
start "" http://localhost:8000
echo [4/4] Done!
echo.
echo   Open http://localhost:8000
echo   API docs: http://localhost:8000/docs
echo.
python -m uvicorn server:app --host 0.0.0.0 --port 8000
