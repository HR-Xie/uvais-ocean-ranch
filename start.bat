@echo off
chcp 65001 >nul
title UVAIS 海洋牧场智能中枢

echo ==========================================
echo   UVAIS 海洋牧场智能中枢 - 一键启动
echo ==========================================
echo.

REM --- 检查 .env ---
if not exist .env (
    echo [1/4] 创建 .env 配置文件...
    copy .env.example .env >nul
    echo   ! 请编辑 .env 填入 DEEPSEEK_API_KEY 后重新运行
    pause
    exit /b 1
)

REM --- 启动数据库 ---
echo [1/4] 启动 Milvus 向量数据库...
docker-compose up -d
echo   等待 Milvus 就绪...
timeout /t 5 /nobreak >nul

REM --- 构建知识库 ---
echo [2/4] 检查知识库状态...
python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://localhost:19530'); exit(0 if c.has_collection('marine_local_hybrid_knowledge') else 1)" >nul 2>&1
if errorlevel 1 (
    echo   首次运行，构建知识库...
    set PYTHONPATH=.
    python rag/milvus_service.py
) else (
    echo   知识库已就绪
)

REM --- 启动应用 ---
echo [3/4] 启动 Streamlit 应用...
start "" http://localhost:8501
echo [4/4] 启动完成！
echo.
echo   打开 http://localhost:8501 使用系统
echo.
streamlit run app.py --server.port 8501
