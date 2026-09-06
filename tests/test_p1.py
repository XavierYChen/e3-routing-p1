"""P1 telemetry, dashboard, and statistics contract tests."""

import json

import pytest
from e3_routing_p0 import build_record

from e3_routing_p1 import LiveDashboard, TrainingTelemetry
from scripts.benchmark_training import bootstrap_ci
from scripts.train_worker import clear_runtime_snapshots


def sample_record(layer="model.1"):
    return build_record(
        run_id="test",
        family="mot",
        layer_name=layer,
        module_type="FakeMoT",
        num_experts=2,
        top_k=1,
        expert_usage=[0.25, 0.75],
        source="last_routing_snapshot",
        mode="train",
    )


def test_dashboard_appends_history_and_replaces_latest(tmp_path):
    dashboard = LiveDashboard(tmp_path)
    dashboard.publish([sample_record("first")], run_id="test", step=0, updated_utc="now")
    dashboard.publish([sample_record("second")], run_id="test", step=1, updated_utc="later")
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["step"] == 1
    assert len(latest["records"]) == 2
    assert len((tmp_path / "routing_records.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert "setInterval(refresh,1000)" in (tmp_path / "dashboard.html").read_text(encoding="utf-8")


def test_dashboard_rejects_invalid_record(tmp_path):
    value = sample_record()
    value["usage_semantics"] = "wrong"
    with pytest.raises(ValueError):
        LiveDashboard(tmp_path).publish([value], run_id="test", step=0, updated_utc="now")


def test_sampling_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        TrainingTelemetry(family="mot", run_id="test", output=tmp_path, enabled=True, sample_every=0)


def test_bootstrap_is_deterministic_and_ordered():
    first = bootstrap_ci([1.01, 1.02, 1.03], seed=7, samples=200)
    second = bootstrap_ci([1.01, 1.02, 1.03], seed=7, samples=200)
    assert first == second
    assert 1.0 < first[0] <= first[1] < 1.1


def test_runtime_snapshot_cleanup_removes_non_leaf_cache():
    class Module:
        pass

    cached = Module()
    cached.last_routing_snapshot = {"weights": object()}
    cached._last_routing_logits = object()
    cached._last_routing_probs = object()
    cached._last_routing_summary = object()
    untouched = Module()

    class InnerModel:
        @staticmethod
        def modules():
            return [cached, untouched]

    class Model:
        model = InnerModel()

    assert clear_runtime_snapshots(Model()) == 4
    assert cached.last_routing_snapshot == {}
    assert cached._last_routing_logits is None
    assert cached._last_routing_probs is None
    assert cached._last_routing_summary is None
    assert not hasattr(untouched, "last_routing_snapshot")
