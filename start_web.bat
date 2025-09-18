@echo off

REM Set UTF-8 encoding
chcp 65001 > nul

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python environment not found. Please install Python first.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found. The program may not run correctly.
    pause
)

REM Start Web service
echo Starting Personal AI Q&A Robot Web Interface...
echo After the service starts, please visit http://localhost:5000 in your browser
echo Press Ctrl+C to stop the service.

python -m src.web_interface

pause