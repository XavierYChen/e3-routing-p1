@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\AI\envs\yolo_master\python.exe"
set "COMMON=--yolo-root D:\AI\YOLO-Master --weights D:\AI\YOLO-Master\yolo26n.pt --data D:\AI\E3-Routing-P2\configs\coco8-local.yaml --project D:\AI\tmp\e3-specialization-final --epochs 10 --imgsz 640 --batch 1 --device 0 --seed 0"
"%PYTHON%" scripts\train_specialization.py --family mot --name mot-pretrained-10e-640 --summary results\trained-routing-20260909\mot-summary.json %COMMON%
if errorlevel 1 exit /b %errorlevel%
"%PYTHON%" scripts\train_specialization.py --family moa --name moa-pretrained-10e-640 --summary results\trained-routing-20260909\moa-summary.json %COMMON%
endlocal
