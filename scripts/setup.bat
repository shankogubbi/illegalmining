@echo off
setlocal
cd /d "%~dp0.."
if not exist "venv\Scripts\python.exe" (
    py -3 -m venv venv
    if errorlevel 1 goto :error
)
"venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 goto :error
echo Setup complete. Run run.bat to open the app or scripts\test.bat to run tests.
exit /b 0
:error
echo Setup failed. Check that Python and pip are installed and review the error above.
exit /b 1
