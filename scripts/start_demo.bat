@echo off
setlocal
cd /d "%~dp0.."
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe scripts\demo_server.py
) else (
    python scripts\demo_server.py
)
pause
