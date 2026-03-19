@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set ROOT=%~dp0
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\greenpower_demo\train.py" %*
