@echo off
rem start.cmd of the Windows package (scripts/windows/package.ps1): settings.ini next to it, then the weight download (23 GB on
rem the first start) and the server in this window, with the OpenAI / Anthropic API and the web UI on http://127.0.0.1:8000/ui.
rem docs/configuration.md, "Windows package".
setlocal
cd /d "%~dp0"
python\python.exe app\scripts\start_package.py
if errorlevel 1 pause
