"""Ultralytics callbacks that stream P0 records and measure full batch cost."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e3_routing_p0 import RoutingCollector

from .dashboard import LiveDashboard


class TrainingTelemetry:
    """Attach reusable P0 collection to a trainer without editing its forward method."""

    def __init__(
        self,
        *,
        family: str,
        run_id: str,
        output: str | Path,
        enabled: bool,
        sample_every: int = 1,
    ) -> None:
        if sample_every < 1:
            raise ValueError("sample_every must be positive")
        self.family, self.run_id = family, run_id
        self.enabled, self.sample_every = enabled, sample_every
        self.dashboard = LiveDashboard(output) if enabled else None
        self.collector: RoutingCollector | None = None
        self.step = 0
        self.started_at = 0.0
        self.batch_ms: list[float] = []
        self.losses: list[float] = []
        self.batch_fingerprints: list[str] = []
        self.records_written = 0

    def register(self, model: Any) -> None:
        """Register the callback surface exposed by Ultralytics Model."""
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_train_batch_start", self.on_train_batch_start)
        model.add_callback("on_train_batch_end", self.on_train_batch_end)
        model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        model.add_callback("on_model_save", self.on_model_save)
        model.add_callback("on_train_end", self.on_train_end)
        model.add_callback("teardown", self.teardown)

    def on_train_start(self, trainer: Any) -> None:
        """Keep hooks batch-scoped so checkpoint serialization never sees them."""

    def on_train_batch_start(self, trainer: Any) -> None:
        self.started_at = time.perf_counter()
        digest = hashlib.sha256()
        for key in sorted(trainer.batch):
            value = trainer.batch[key]
            digest.update(str(key).encode())
            if hasattr(value, "detach"):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            elif key == "im_file":
                digest.update("\n".join(map(str, value)).encode())
        self.batch_fingerprints.append(digest.hexdigest())
        if self.enabled and self.step % self.sample_every == 0:
            self._close()
            self.collector = RoutingCollector(
                trainer.model,
                run_id=self.run_id,
                families=(self.family,),
                step=self.step,
                mode="train",
            )
            self.collector.__enter__()

    def on_train_batch_end(self, trainer: Any) -> None:
        try:
            if self.collector:
                records = self.collector.records[:]
                assert self.dashboard is not None
                self.dashboard.publish(
                    records,
                    run_id=self.run_id,
                    step=self.step,
                    updated_utc=datetime.now(UTC).isoformat(),
                )
                self.records_written += len(records)
            loss = getattr(trainer, "loss", None)
            if loss is not None:
                self.losses.append(float(loss.detach().cpu().item() if hasattr(loss, "detach") else loss))
        finally:
            self._close()
            self.batch_ms.append((time.perf_counter() - self.started_at) * 1000)
            self.step += 1

    def _close(self) -> None:
        if self.collector:
            self.collector.__exit__(None, None, None)
            self.collector = None

    def on_train_end(self, _trainer: Any) -> None:
        self._close()

    def on_train_epoch_end(self, _trainer: Any) -> None:
        self._close()

    def on_model_save(self, _trainer: Any) -> None:
        self._close()

    def teardown(self, _trainer: Any) -> None:
        self._close()
