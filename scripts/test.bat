@echo off
setlocal
cd /d "%~dp0.."
if not exist "venv\Scripts\python.exe" (
    echo Run scripts\setup.bat first.
    exit /b 1
)
"venv\Scripts\python.exe" -m pytest tests %*
exit /b %errorlevel%
