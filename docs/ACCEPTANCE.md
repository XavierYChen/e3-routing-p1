# P1 验收

- [x] 复用 P0 的统一 schema 和采集器。
- [x] MoE、MoT、Latent 实时面板。
- [x] JSONL 历史与原子 latest 快照。
- [x] 正式训练 batch on/off 计时口径。
- [x] AB/BA 顺序、warm-up、重复和 bootstrap 95% 区间。
- [x] 真实 COCO8 三族基准中位减速均 <10%。
- [x] 保存完整结果、面板截图并提交至 GitHub。

## 验收数字

证据目录：`results/verified-p1-cpu-v1-20260906`

| 路由族 | 配对比值（3 次） | 中位减速 | 采集记录 |
|---|---|---:|---:|
| MoE | 1.0474, 1.0434, 0.9895 | +4.34% | 90 |
| MoT | 0.9575, 1.0101, 1.1426 | +1.01% | 60 |
| Latent | 0.9981, 1.0057, 1.0673 | +0.57% | 45 |

18 个 worker 均通过初始模型 SHA-256、输入 batch 指纹和逐 batch loss 一致性检查。开启采集的时间包括 forward hook、schema 校验、JSONL 追加以及原子 `latest.json` 更新。
