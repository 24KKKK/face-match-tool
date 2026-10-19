@echo off
chcp 437 >nul 2>nul
title face-match-tool
cd /d "%~dp0"

set "PY=D:\myprogram\Python\Python312\python.exe"
if exist "%PY%" goto RUN

set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py

:RUN
echo Starting face-match-tool ...
echo Browser will open at http://127.0.0.1:8777/
echo Keep this window open. Closing it will stop the tool.
echo.
"%PY%" webui.py

if errorlevel 1 (
  echo.
  echo Failed to start. Please make sure Python with insightface is available.
)
pause
