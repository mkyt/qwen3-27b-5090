@echo off
rem update.cmd of the Windows package (scripts/windows/package.ps1): updates this folder to the latest release, or to
rem "update.cmd -Tag v0.3.1", with a copy of app\scripts\update.ps1. settings.ini, the weights and the log stay.
rem "update.cmd -Check" only shows the versions. docs/configuration.md, "Updating".
rem (One line: the update replaces this file, and cmd reads a file line by line as it runs it. A copy of the script in
rem %TEMP%: PowerShell keeps the file it runs open, which would keep app\ from being moved.)
copy /y "%~dp0app\scripts\update.ps1" "%TEMP%\qwen3-27b-5090-update.ps1" >nul && powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\qwen3-27b-5090-update.ps1" -Root "%~dp0." %* & pause & exit /b
