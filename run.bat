@echo off
rem seq2formula: start the local server and open the dashboard.
cd /d "%~dp0"
start /b "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000/"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
