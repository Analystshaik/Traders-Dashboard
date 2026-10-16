@echo off
setlocal
cd /d "%~dp0"
set TASK_NAME=TLT Trader EOD Refresh
set RUNNER=%CD%\run_eod_refresh.bat
schtasks /Create /TN "%TASK_NAME%" /TR "\"%RUNNER%\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 17:15 /F
echo.
echo Task created for Monday-Friday at 5:15 PM local Windows time.
echo TWS or IB Gateway must already be running and logged in.
pause
