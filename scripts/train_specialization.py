"""Train a routed checkpoint for downstream P2 analysis and archive its provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--yolo-root", type=Path, default=ROOT.parent / "YOLO-Master")
    result.add_argument("--family", choices=("mot", "moa"), required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--project", type=Path, required=True)
    result.add_argument("--name", required=True)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--epochs", type=int, default=10)
    result.add_argument("--imgsz", type=int, default=640)
    result.add_argument("--batch", type=int, default=2)
    result.add_argument("--device", default="0")
    result.add_argument("--seed", type=int, default=0)
    return result


def main() -> int:
    args = parser().parse_args()
    yolo_root = args.yolo_root.resolve()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(yolo_root))
    config_dir = ROOT / "env/runtime"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import SETTINGS

    def clear_runtime_snapshots(owner) -> int:
        cleared = 0
        for module in owner.model.modules():
            if hasattr(module, "last_routing_snapshot"):
                module.last_routing_snapshot = {}
                cleared += 1
            for name in ("_last_routing_logits", "_last_routing_probs", "_last_routing_summary"):
                if hasattr(module, name):
                    setattr(module, name, None)
                    cleared += 1
        return cleared

    class SnapshotSafeDetectionTrainer(DetectionTrainer):
        def setup_model(self):
            checkpoint = super().setup_model()
            clear_runtime_snapshots(self.model)
            return checkpoint

    datasets = ROOT.parent / "datasets"
    if datasets.is_dir():
        SETTINGS.update({"datasets_dir": str(datasets)})
    profiles = {"mot": "yolo26-master-mot-n.yaml", "moa": "yolo26-master-moa-n.yaml"}
    yaml_path = yolo_root / "ultralytics/cfg/models/26" / profiles[args.family]
    weights = args.weights.resolve()
    model = YOLO(yaml_path)
    model.load(weights)
    clear_runtime_snapshots(model)

    def clear_trainer_snapshots(trainer):
        clear_runtime_snapshots(trainer.model)

    for event in ("on_train_batch_end", "on_train_epoch_end", "on_model_save", "on_train_end"):
        model.add_callback(event, clear_trainer_snapshots)
    model.train(
        trainer=SnapshotSafeDetectionTrainer,
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,
        amp=True,
        val=True,
        plots=True,
        save=True,
        seed=args.seed,
        deterministic=True,
        cache=False,
        scale=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        close_mosaic=3,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        verbose=False,
    )
    run_dir = args.project.resolve() / args.name
    best = run_dir / "weights/best.pt"
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    metric = "metrics/mAP50-95(B)"
    best_row = max(rows, key=lambda row: float(row[metric]))
    published = args.summary.resolve().parent
    published.mkdir(parents=True, exist_ok=True)
    for filename in ("results.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg"):
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, published / f"{args.family}-{filename}")
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "family": args.family.upper(),
        "model_yaml": str(yaml_path),
        "base_weights_sha256": sha256(weights),
        "checkpoint_local_only": str(best),
        "checkpoint_sha256": sha256(best),
        "checkpoint_published": False,
        "config": {
            "data": str(args.data.resolve()),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "seed": args.seed,
            "amp": True,
            "scale": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.1,
        },
        "best_epoch": int(float(best_row["epoch"])),
        "metrics": {key.strip(): float(value) for key, value in best_row.items() if key.strip() != "epoch"},
        "interpretation": "COCO8 is a four-image functional dataset and overlaps the COCO pretraining domain; metrics are a pipeline check, not an independent generalization claim.",
    }
    args.summary.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
