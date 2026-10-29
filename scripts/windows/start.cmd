@echo off
rem start.cmd of the Windows package (scripts/windows/package.ps1): downloads the weights on the first start (23 GB,
rem into HF_HOME, by default %USERPROFILE%\.cache\huggingface), then serves the OpenAI / Anthropic API and the web UI
rem (http://127.0.0.1:8000/ui). Settings are environment variables (docs/configuration.md); HOST (default 127.0.0.1,
rem 0.0.0.0 for the network), PORT (8000) and OPEN_UI (1: open the web UI in the browser; 0: not) here.
setlocal
cd /d "%~dp0"
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8000"
python\python.exe app\scripts\download_weights.py
if errorlevel 1 (
  echo The weights could not be downloaded.
  pause
  exit /b 1
)
if not "%OPEN_UI%"=="0" start "" "http://127.0.0.1:%PORT%/ui"
python\python.exe app\frontend\server.py --host %HOST%
if errorlevel 1 pause
