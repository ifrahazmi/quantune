@echo off
setlocal enabledelayedexpansion
title Quantune Launcher

cd /d "%~dp0"

echo ==================================================
echo                  Starting Quantune
echo ==================================================
echo.

:: 1. Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Setting it up...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
        pause
        exit /b 1
    )
    echo [INFO] Installing dependencies...
    ".venv\Scripts\pip.exe" install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.
        pause
        exit /b 1
    )
)

:: 2. Check if port 8000 is already occupied and free it to prevent WinError 10013
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Port 8000 is in use by PID %%a. Freeing port...
    taskkill /F /PID %%a >nul 2>&1
)

:: 3. Automatically open browser after 2 seconds
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

:: 4. Start Quantune with Uvicorn
echo [INFO] Quantune is running!
echo [INFO] Dashboard: http://localhost:8000
echo [INFO] Press Ctrl+C in this window to stop the server.
echo.
".venv\Scripts\python.exe" -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error.
    pause
)
