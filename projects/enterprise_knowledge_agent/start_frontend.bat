@echo off
chcp 65001 >nul
rem Switch to agent-learning so that "projects.xxx" is importable.
cd /d "%~dp0..\.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m projects.enterprise_knowledge_agent.scripts.run_frontend %*
) else (
    python -m projects.enterprise_knowledge_agent.scripts.run_frontend %*
)

echo.
echo Frontend process ended. Press any key to close.
pause >nul
