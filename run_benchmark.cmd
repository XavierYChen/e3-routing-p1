@echo off
setlocal
cd /d "%~dp0"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
if not defined E3_YOLO_MASTER_ROOT set "E3_YOLO_MASTER_ROOT=%~dp0..\YOLO-Master"
if not defined E3_P0_ROOT set "E3_P0_ROOT=%~dp0..\E3-Routing-P0"
if not defined E3_PYTHON set "E3_PYTHON=python"
"%E3_PYTHON%" scripts\benchmark_training.py %*
exit /b %errorlevel%

