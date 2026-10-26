@echo off
rem seq2formula: start the local server and open the dashboard.
cd /d "%~dp0"

rem Kill any process still listening on port 8002 from a previous run.
rem Without this, uvicorn fails with Errno 10048 and the browser keeps
rem talking to the stale server (old Python code, fresh static assets).
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8002 ^| findstr LISTENING') do (
    echo Stopping old server process %%a...
    taskkill /F /PID %%a >nul 2>&1
)

rem Give the OS a moment to release the port before uvicorn binds it.
timeout /t 1 /nobreak >nul

start /b "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8002/"
python -m uvicorn app:app --host 127.0.0.1 --port 8002
