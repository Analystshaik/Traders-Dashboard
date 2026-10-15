@echo off
setlocal

echo TLT Trader's Dashboard - Copy Cloud Database
set /p SOURCE_DB=Enter full path to your working SQLite database: 

if not exist "%SOURCE_DB%" (
    echo ERROR: File not found.
    pause
    exit /b 1
)

if not exist data mkdir data
copy /Y "%SOURCE_DB%" "data\streamlit_market_data.sqlite"

echo.
python validate_cloud_db.py
pause
