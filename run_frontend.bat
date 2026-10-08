@echo off
title Smart Education - Streamlit Frontend Dashboard
cd /d "%~dp0"
echo ======================================================================
echo   Launching Smart Education Interactive Frontend Dashboard...
echo ======================================================================
echo Execution Device: NVIDIA RTX 3050 GPU (with CUDA PyTorch)
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Streamlit encountered an issue while running.
    pause
)
