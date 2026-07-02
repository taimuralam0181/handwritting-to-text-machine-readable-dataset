@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
)
echo Installing API dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Starting Prescription API on http://127.0.0.1:8001
echo Keep this window open while using the mobile app.
echo.
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001
pause
