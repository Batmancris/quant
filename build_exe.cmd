@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set ROOT=%~dp0
"%ROOT%\.venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean "%ROOT%quant_desktop.spec"
