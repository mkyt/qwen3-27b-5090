@echo off
rem Start the server from Windows: double-click this file, or make a shortcut to it (on the desktop; in shell:startup
rem to start with the logon). It runs scripts/serve.sh in the WSL distro this checkout is in, opens the stats page and
rem keeps the distro running while the container runs (WSL shuts an idle distro down, and Docker with it).
rem   serve.cmd           start; the engine loads with the compose.yaml settings
rem   serve.cmd idle      start the server only (ENGINE_AUTOSTART=0: no VRAM until the engine is started at /ui)
rem   serve.cmd stop      stop the container
rem   serve.cmd status
setlocal
set "HERE=%~dp0"
set "P=%HERE:\\wsl.localhost\=%"
set "P=%P:\\wsl$\=%"
if "%P%"=="%HERE%" (
  echo Run this file from the checkout in WSL, e.g. \\wsl.localhost\Ubuntu\home\you\qwen3-27b-5090\scripts\windows\serve.cmd
  pause
  exit /b 1
)
for /f "tokens=1* delims=\" %%a in ("%P%") do (
  set "DISTRO=%%a"
  set "DIR=%%b"
)
set "DIR=/%DIR:\=/%../.."

set "ARGS=%~1"
if "%ARGS%"=="" set "ARGS=up"
if /i "%ARGS%"=="idle" set "ARGS=up --idle"
if /i "%ARGS%"=="stop" set "ARGS=down"

wsl.exe -d %DISTRO% --cd "%DIR%" -- bash scripts/serve.sh %ARGS%
if errorlevel 1 (
  pause
  exit /b 1
)
if /i "%ARGS:~0,2%"=="up" (
  powershell.exe -NoProfile -Command "Start-Process wsl.exe -WindowStyle Hidden -ArgumentList '-d','%DISTRO%','--cd','%DIR%','--','bash','scripts/serve.sh','keepalive'"
  start "" http://127.0.0.1:8000/ui
)
if /i "%ARGS%"=="status" pause
