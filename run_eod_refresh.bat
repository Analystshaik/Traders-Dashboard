@echo off
setlocal
cd /d "%~dp0"
python eod_refresh_and_push.py
exit /b %ERRORLEVEL%
