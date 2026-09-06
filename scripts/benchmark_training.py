"""Run paired COCO8 training with routing telemetry off/on for three families."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 4000) -> list[float]:
    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def render_overhead(families: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(families)
    slowdowns = [families[name]["median_slowdown_pct"] for name in names]
    lower = [(families[name]["median_ratio"] - families[name]["bootstrap_95_ci_ratio"][0]) * 100 for name in names]
    upper = [(families[name]["bootstrap_95_ci_ratio"][1] - families[name]["median_ratio"]) * 100 for name in names]
    colors = ["#16a34a" if value < 10 else "#dc2626" for value in slowdowns]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bars = axis.bar(names, slowdowns, color=colors, width=0.62)
    axis.errorbar(names, slowdowns, yerr=[lower, upper], fmt="none", color="#111827", capsize=5)
    axis.axhline(10, color="#d97706", linestyle="--", linewidth=1.5, label="P1 threshold: 10%")
    axis.axhline(0, color="#64748b", linewidth=0.8)
    axis.set(title="E3 P1 paired COCO8 training overhead", ylabel="Median slowdown (%)")
    axis.legend(loc="upper right")
    for bar, value in zip(bars, slowdowns):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.2f}%", ha="center", va="bottom")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_dashboard_snapshot(p0_root: Path, benchmark_output: Path) -> None:
    sys.path.insert(0, str(p0_root))
    from e3_routing_p0 import render_static

    records = []
    for family in ("moe", "mot", "latent"):
        latest = benchmark_output / "dashboard" / family / "rep-0/latest.json"
        records.extend(json.loads(latest.read_text(encoding="utf-8"))["records"])
    render_static(records, benchmark_output / "dashboard_snapshot.png", title="E3 P1 live dashboard snapshot")


def write_manifest(output: Path) -> Path:
    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.sha256.json"):
        relative = str(path.relative_to(output)).replace("\\", "/")
        files[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    destination = output / "manifest.sha256.json"
    destination.write_text(json.dumps({"algorithm": "sha256", "files": files}, indent=2) + "\n", encoding="utf-8")
    return destination


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--yolo-root", type=Path, default=Path(os.environ.get("E3_YOLO_MASTER_ROOT", ROOT.parent / "YOLO-Master"))
    )
    result.add_argument(
        "--p0-root", type=Path, default=Path(os.environ.get("E3_P0_ROOT", ROOT.parent / "E3-Routing-P0"))
    )
    result.add_argument(
        "--output", type=Path, default=ROOT / "results" / datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    )
    result.add_argument("--data", default="coco8.yaml")
    result.add_argument("--epochs", type=int, default=5)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--imgsz", type=int, default=64)
    result.add_argument("--batch", type=int, default=2)
    result.add_argument("--device", default="cpu")
    result.add_argument("--warmup-batches", type=int, default=1)
    result.add_argument("--sample-every", type=int, default=2)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--keep-train-artifacts",
        action="store_true",
        help="Keep framework checkpoints and CSV files from all paired workers (can exceed 1 GB).",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.yolo_root, args.p0_root, args.output = args.yolo_root.resolve(), args.p0_root.resolve(), args.output.resolve()
    for path, label in ((args.yolo_root, "YOLO-Master"), (args.p0_root, "P0")):
        if not path.is_dir():
            raise SystemExit(f"{label} root not found: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = {
        "moe": "yolo26-master-n.yaml",
        "mot": "yolo26-master-mot-n.yaml",
        "latent": "yolo26-master-latent-n.yaml",
    }
    raw_runs = []
    ratios: dict[str, list[float]] = {family: [] for family in profiles}
    for family in profiles:
        for repetition in range(args.repetitions):
            pair: dict[str, float] = {}
            order = ("off", "on") if repetition % 2 == 0 else ("on", "off")
            for condition in order:
                run_id = f"{family}-rep{repetition}-{condition}"
                run_seed = args.seed + repetition
                result_path = args.output / "worker-results" / f"{run_id}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(ROOT / "scripts/train_worker.py"),
                    "--yolo-root",
                    str(args.yolo_root),
                    "--p0-root",
                    str(args.p0_root),
                    "--result",
                    str(result_path),
                    "--dashboard",
                    str(args.output / "dashboard" / family / f"rep-{repetition}"),
                    "--project",
                    str(args.output / "train-runs"),
                    "--family",
                    family,
                    "--condition",
                    condition,
                    "--run-id",
                    run_id,
                    "--data",
                    args.data,
                    "--epochs",
                    str(args.epochs),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--device",
                    args.device,
                    "--sample-every",
                    str(args.sample_every),
                    "--seed",
                    str(run_seed),
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                log_path = args.output / "logs" / f"{run_id}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"{run_id} failed; see {log_path}")
                run = json.loads(result_path.read_text(encoding="utf-8"))
                measured = run["batch_ms"][args.warmup_batches :]
                if not measured:
                    raise RuntimeError(f"{run_id}: no batches remain after warm-up")
                mean_ms = statistics.mean(measured)
                pair[condition] = mean_ms
                raw_runs.append(
                    {
                        **run,
                        "repetition": repetition,
                        "measured_batch_ms": measured,
                        "mean_batch_ms": mean_ms,
                        "median_batch_ms": statistics.median(measured),
                    }
                )
                print(f"[{family} rep={repetition} {condition}] mean={mean_ms:.3f} ms", flush=True)
            ratios[family].append(pair["on"] / pair["off"])
            pair_runs = {
                run["condition"]: run for run in raw_runs if run["family"] == family and run["repetition"] == repetition
            }
            off_losses, on_losses = pair_runs["off"]["losses"], pair_runs["on"]["losses"]
            if pair_runs["off"]["initial_model_sha256"] != pair_runs["on"]["initial_model_sha256"]:
                raise RuntimeError(f"{family} repetition {repetition}: initial model fingerprints differ")
            if pair_runs["off"]["batch_fingerprints"] != pair_runs["on"]["batch_fingerprints"]:
                raise RuntimeError(f"{family} repetition {repetition}: training batch fingerprints differ")
            if len(off_losses) != len(on_losses) or any(
                not math.isclose(off, on, rel_tol=1e-5, abs_tol=1e-6) for off, on in zip(off_losses, on_losses)
            ):
                raise RuntimeError(f"{family} repetition {repetition}: telemetry changed the training loss")

    families = {}
    for index, family in enumerate(profiles):
        family_ratios = ratios[family]
        ratio = statistics.median(family_ratios)
        ci = bootstrap_ci(family_ratios, seed=args.seed + index)
        on_runs = [run for run in raw_runs if run["family"] == family and run["condition"] == "on"]
        families[family] = {
            "paired_mean_ratios": family_ratios,
            "median_ratio": ratio,
            "median_slowdown_pct": (ratio - 1.0) * 100,
            "bootstrap_95_ci_ratio": ci,
            "records_written": sum(run["records_written"] for run in on_runs),
            "passed_median_under_10_percent": ratio < 1.10,
        }
    passed = all(item["passed_median_under_10_percent"] and item["records_written"] > 0 for item in families.values())
    summary = {
        "status": "passed" if passed else "failed",
        "scope": "paired real COCO8 training batch time including routing collection and live dashboard writes",
        "families": families,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "limitations": [
            "CPU laptop microbenchmark; report confidence interval and raw batches.",
            "Randomly initialized models test infrastructure overhead, not detection quality.",
        ],
    }
    (args.output / "raw_runs.json").write_text(json.dumps(raw_runs, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render_overhead(families, args.output / "training_overhead.png")
    render_dashboard_snapshot(args.p0_root, args.output)
    if not args.keep_train_artifacts:
        shutil.rmtree(args.output / "train-runs", ignore_errors=True)
    write_manifest(args.output)
    print(f"Result: {summary['status']} | evidence={args.output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
