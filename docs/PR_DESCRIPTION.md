# 改动摘要

在既有 P1 实时面板和配对基准上增加预训练 GPU 重跑协议：从腾讯 `yolo26n.pt` 迁移权重，以 COCO8、640px、10 epochs、3 seed 比较 telemetry on/off；另提供 snapshot-safe 的 MOT/MOA checkpoint 训练入口供 P2 使用。修复 P0 对合法 float16 概率和舍入误差过严的问题，未修改腾讯 forward。

# 测试证据

- `python -m pytest -q`：P1 5/5 通过；P0 11/11 通过。
- 正式目录：`results/verified-p1-gpu-pretrained-10e-20260909`。
- 三族共 18 个隔离 worker；同一配对的初始权重 SHA-256 与全部 batch 指纹一致。
- checkpoint 训练与验证均完成，MOT/MOA 权重可被 P2 重新加载；公开 SHA-256，不上传权重本体。

# 消融数据

| 家族 | telemetry 中位减速 | 95% bootstrap 区间 | 结论 |
|---|---:|---:|---|
| MOE | -0.95% | -1.84%～+3.31% | PASS |
| MOT | +2.15% | -0.06%～+9.62% | PASS |
| LATENT | +1.66% | -1.53%～+3.02% | PASS |

本轮不是正式专家分工消融。训练后 MOT/MOA 单 seed 的 mAP50-95 分别为 0.03434/0.03750，仅证明链路有效。

# 已知局限

- COCO8 仅 4 张训练图，且与 COCO 预训练域重合，不能支持泛化精度结论。
- MOT CUDA 训练存在跨进程非确定性；off/off 平均相对 loss 漂移 2.33%、最大 12.23%。
- 开销基准采用 FP32、batch=1、每 8 batch 采样；每 2 batch 采样的 MOE 首组超过 10%。
- checkpoint 目前只有 seed 0；多 seed 专门化实验留给后续正式消融。
