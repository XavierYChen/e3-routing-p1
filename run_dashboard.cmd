@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_dashboard.cmd results\run-directory\dashboard\family\rep-number
  exit /b 2
)
if not defined E3_PYTHON set "E3_PYTHON=python"
"%E3_PYTHON%" -m e3_routing_p1.serve "%~1"
exit /b %errorlevel%

