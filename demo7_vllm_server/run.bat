@echo off
REM Quick start script for TinyLlama Inference Server
REM Windows-compatible version (no vLLM C extension issues)

echo.
echo ⚡ TinyLlama Inference Server - Quick Start
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if requirements are installed
python -c "import torch; import transformers" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
)

echo.
echo 🚀 Starting TinyLlama server on http://127.0.0.1:8001
echo.
echo First time? Model download will start (~2-3 minutes, ~2.5GB)
echo.

uvicorn vllm_server_transformers:app --reload --host 127.0.0.1 --port 8001

pause
