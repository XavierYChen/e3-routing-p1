@echo off
setlocal
cd /d "%~dp0"
set "E3_YOLO_MASTER_ROOT=D:\AI\YOLO-Master"
set "E3_P0_ROOT=D:\AI\E3-Routing-P0"
set "E3_PYTHON=D:\AI\envs\yolo_master\python.exe"
"%E3_PYTHON%" scripts\benchmark_training.py ^
  --yolo-root "%E3_YOLO_MASTER_ROOT%" ^
  --p0-root "%E3_P0_ROOT%" ^
  --output results\verified-p1-gpu-pretrained-10e-20260909 ^
  --data D:\AI\E3-Routing-P2\configs\coco8-local.yaml ^
  --epochs 10 --repetitions 3 --imgsz 640 --batch 1 --device 0 ^
  --warmup-batches 4 --sample-every 8 --seed 0 --allow-loss-drift ^
  --pretrained-weights D:\AI\YOLO-Master\yolo26n.pt
endlocal
