# E3 Routing Observer · P1

P1 在 [P0 统一采集库](https://github.com/XavierYChen/e3-routing-p0) 上增加两项验收能力：

1. 不依赖云服务的本地实时路由面板，覆盖 MoE、MoT、Latent；
2. 真实 COCO8 训练 on/off、AB/BA 交替、重复测量和置信区间，门槛为中位减速 <10%。

阶段导航：[Smoke](https://github.com/XavierYChen/e3-routing-smoke) · [P0](https://github.com/XavierYChen/e3-routing-p0) · **P1（本仓库）** · P2 token 原图叠加与演示（计划）。

实时面板使用浏览器原生 HTML/JavaScript，每秒读取原子更新的 `latest.json`。训练线程同时追加 `routing_records.jsonl`，断电前的历史仍然保留。没有修改腾讯核心 forward。

## 已验证结果

2026-09-06 在 Windows 11、Intel i7-12700H CPU 上完成真实 COCO8 配对训练。每族运行 3 组，每组分别关闭和开启采集；每次 5 epochs。三族的中位减速均低于 P1 的 10% 门槛。

| 路由族 | 中位减速 | bootstrap 95% 区间 | 写入记录 | 结论 |
|---|---:|---:|---:|---|
| MoE | +4.34% | -1.05% ～ +4.74% | 90 | 通过 |
| MoT | +1.01% | -4.25% ～ +14.26% | 60 | 通过 |
| Latent | +0.57% | -0.19% ～ +6.73% | 45 | 通过 |

![P1 paired training overhead](results/verified-p1-cpu-v1-20260906/training_overhead.png)

![P1 dashboard snapshot](results/verified-p1-cpu-v1-20260906/dashboard_snapshot.png)

完整证据位于 [`results/verified-p1-cpu-v1-20260906`](results/verified-p1-cpu-v1-20260906)：包含原始批次时间、18 个独立 worker 结果、训练日志、实时面板数据、静态图和 SHA-256 清单。MoT 的区间跨过 10%，说明三次 CPU 重复对它的区间估计仍较宽；验收结论按任务要求使用中位减速。

## 运行基准

```bat
cd /d D:\AI\E3-Routing-P1
set "E3_YOLO_MASTER_ROOT=D:\AI\YOLO-Master"
set "E3_P0_ROOT=D:\AI\E3-Routing-P0"
set "E3_PYTHON=D:\AI\envs\yolo_master\python.exe"
run_benchmark.cmd
```

打开某次 on 运行的面板：

```bat
run_dashboard.cmd results\你的运行目录\dashboard\moe\rep-0
```

浏览器访问 `http://127.0.0.1:8765/dashboard.html`。服务器只监听本机。

测试采集和面板逻辑：

```bat
run_tests.cmd
```

## 口径

- 每族 3 个重复，顺序为 off/on、on/off、off/on。
- 每次 5 epochs，batch=2，imgsz=64，CPU，FP32，workers=0。
- 每次丢弃第一个 batch 作为 warm-up。
- 默认每 2 个 batch 采集一次；on 的平均 batch 时间包含 hook 采集、校验、JSONL 追加和 `latest.json` 更新。
- 先计算每个重复的 on/off 平均 batch 时间比值，再报告三个重复的中位减速和 bootstrap 95% 区间。
- 每个条件在独立 Python 进程中运行；相同重复的初始模型、输入 batch 和逐 batch loss 必须完全配对，否则基准直接失败。
- 默认删除框架生成的 1 GB 以上临时 checkpoint；如确需保留，可传入 `--keep-train-artifacts`。

实现范围、测量解释和已知问题见 [`limitations.md`](limitations.md)，验收逐项证据见 [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)。
