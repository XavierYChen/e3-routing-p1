# P1 预训练 GPU 重跑记录

## 问题与口径

本轮验证两个问题：在 640px、10 epochs 和预训练迁移的更真实训练状态下，P1 采集是否仍满足中位减速 `<10%`；训练出的空间路由 checkpoint 是否足以让 P2 分析真实的专家变化。开销和模型效果分开运行，避免验证与保存开销污染 telemetry on/off 数字。

| 项目 | 开销基准 | P2 checkpoint 训练 |
|---|---|---|
| 数据 | COCO8，同一 4 图 | COCO8，同一 4 图 |
| 图像 | 640px | 640px |
| epoch | 10 | 10 |
| batch | 1 | 1 |
| seed | 0/1/2 | 0（后续消融补多 seed） |
| 精度 | FP32 | AMP |
| 增广 | 全关，确保批次严格配对 | scale=.5, mosaic=1, copy_paste=.1 |
| 验证/保存 | 关闭 | 开启 |
| 预训练 | `yolo26n.pt`，SHA-256 `9b09cc…d4fef` | 同左 |

## 最终结果

| 家族 | 中位减速 | 95% bootstrap 区间 | 记录数 | 门槛 |
|---|---:|---:|---:|---|
| MOE | -0.95% | -1.84%～+3.31% | 90 | PASS |
| MOT | +2.15% | -0.06%～+9.62% | 60 | PASS |
| LATENT | +1.66% | -1.53%～+3.02% | 45 | PASS |

MOT/MOA checkpoint 的验证指标分别为 mAP50-95 `0.03434` 和 `0.03750`。这些数字只证明训练、保存、验证和后续加载链路有效。COCO8 太小，且基础权重来自 COCO 域，因此不做精度优越性结论。

## 失败记录与决策

| 触发 | 观察 | 决策 |
|---|---|---|
| AMP on/off | MOT 第 8 batch 后轨迹分叉 | 开销基准切换 FP32；checkpoint 训练保留 AMP |
| FP32 batch=2 | MOE 优化器更新时显存不足 | 三族统一固定 batch=1 |
| 每 2 batch 采样 | MOE 首组约 24% 开销 | 改为每 8 batch，仍保留 5 个采样点 |
| MOT FP32 on/off | loss 仍跨进程分叉 | 增跑 off/off；确认 CUDA 后端本身存在 2.33% 平均、12.23% 最大漂移 |
| checkpoint 保存 | 路由诊断缓存不可序列化 | 在外部 trainer 回调中保存前清缓存，不修改腾讯 forward |

失败输出保留在本机 `D:\AI\tmp\e3-p1-failures`。正式目录只包含最终协议；机器可读 `summary.json` 同时给出每对时间比值和 loss 漂移。

## 局限

- 三个 seed 满足统一实验红线的最低统计要求，但 bootstrap 区间仍宽。
- MOT CUDA 算子不是跨进程逐位确定；本轮不能宣称 hook 对训练轨迹零影响，只能给出同批次配对与 off/off 漂移参照。
- checkpoint 专门化仅单 seed，属于 P2 改进输入；正式专家分工消融将在 P1/P2 完整后另做。
