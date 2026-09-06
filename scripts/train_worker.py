"""Run one isolated training condition for the paired P1 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--yolo-root", type=Path, required=True)
    result.add_argument("--p0-root", type=Path, required=True)
    result.add_argument("--result", type=Path, required=True)
    result.add_argument("--dashboard", type=Path, required=True)
    result.add_argument("--project", type=Path, required=True)
    result.add_argument("--family", choices=("moe", "mot", "latent"), required=True)
    result.add_argument("--condition", choices=("off", "on"), required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--data", default="coco8.yaml")
    result.add_argument("--epochs", type=int, required=True)
    result.add_argument("--imgsz", type=int, required=True)
    result.add_argument("--batch", type=int, required=True)
    result.add_argument("--device", required=True)
    result.add_argument("--sample-every", type=int, default=1)
    result.add_argument("--seed", type=int, required=True)
    return result


def model_fingerprint(model) -> str:
    digest = hashlib.sha256()

    def update(name, value):
        digest.update(str(name).encode())
        if hasattr(value, "detach"):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for child_name, child in sorted(value.items()):
                update(child_name, child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                update(index, child)
        else:
            digest.update(repr(value).encode())

    for name, value in sorted(model.model.state_dict().items()):
        update(name, value)
    return digest.hexdigest()


def clear_runtime_snapshots(model) -> int:
    """Drop diagnostic caches that must not be copied into the trainer EMA model."""
    cleared = 0
    for module in model.model.modules():
        if hasattr(module, "last_routing_snapshot"):
            module.last_routing_snapshot = {}
            cleared += 1
        for name in ("_last_routing_logits", "_last_routing_probs", "_last_routing_summary"):
            if hasattr(module, name):
                setattr(module, name, None)
                cleared += 1
    return cleared


def main() -> int:
    args = parser().parse_args()
    for path in (ROOT, args.p0_root.resolve(), args.yolo_root.resolve()):
        sys.path.insert(0, str(path))
    os.environ["MOE_SNAPSHOT_INTERVAL"] = "1"
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    config_dir = ROOT / "env/runtime"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import SETTINGS

    from e3_routing_p1 import TrainingTelemetry

    class SnapshotSafeDetectionTrainer(DetectionTrainer):
        """Clear model-build diagnostic tensors before upstream creates its EMA copy."""

        def setup_model(self):
            checkpoint = super().setup_model()
            clear_runtime_snapshots(self.model)
            return checkpoint

    local_datasets = args.yolo_root.resolve().parent / "datasets"
    if (local_datasets / "coco8").is_dir():
        SETTINGS.update({"datasets_dir": str(local_datasets)})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    profiles = {
        "moe": "yolo26-master-n.yaml",
        "mot": "yolo26-master-mot-n.yaml",
        "latent": "yolo26-master-latent-n.yaml",
    }
    model = YOLO(str(args.yolo_root / "ultralytics/cfg/models/26" / profiles[args.family]))
    snapshots_cleared = clear_runtime_snapshots(model)
    initial_model_sha256 = model_fingerprint(model)
    telemetry = TrainingTelemetry(
        family=args.family,
        run_id=args.run_id,
        output=args.dashboard,
        enabled=args.condition == "on",
        sample_every=args.sample_every,
    )
    telemetry.register(model)

    def clear_trainer_snapshots(trainer):
        clear_runtime_snapshots(trainer.model)

    # LatentMixture retains graph-connected diagnostics after every forward. Clear them
    # after telemetry has published so upstream recovery/checkpoint copies remain safe.
    model.add_callback("on_train_batch_end", clear_trainer_snapshots)
    model.add_callback("on_train_epoch_end", clear_trainer_snapshots)
    model.add_callback("on_train_end", clear_trainer_snapshots)
    started = time.perf_counter()
    model.train(
        trainer=SnapshotSafeDetectionTrainer,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        amp=False,
        val=False,
        save=False,
        plots=False,
        verbose=False,
        deterministic=True,
        seed=args.seed,
        project=str(args.project),
        name=args.run_id,
        exist_ok=False,
        cache=False,
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
        warmup_epochs=0.0,
        optimizer="SGD",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        erasing=0.0,
    )
    payload = {
        "family": args.family,
        "condition": args.condition,
        "run_id": args.run_id,
        "seed": args.seed,
        "batch_ms": telemetry.batch_ms,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "records_written": telemetry.records_written,
        "losses": telemetry.losses,
        "batch_fingerprints": telemetry.batch_fingerprints,
        "initial_model_sha256": initial_model_sha256,
        "initial_snapshots_cleared": snapshots_cleared,
        "hooks_removed": telemetry.collector is None,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
