import faulthandler

import numpy as np
import pytest

from exp593_dna import anneal_train, anneal_validation
from exp593_dna.anneal_evaluate import scored_base_loss


def smoke():
    return anneal_train.TrainingConfig(
        trial="181b",
        cluster="cw-rno2a",
        nodes=4,
        validation_updates=64,
        run_id="dna-exp593-smoke-watchdog-check",
    )


def test_production_rejected_before_native_run_or_dispatch(monkeypatch):
    def forbidden(_):
        pytest.fail("Production reached the diagnostic wrapper")

    monkeypatch.setattr(anneal_validation, "_NATIVE_RUN", forbidden)
    monkeypatch.setattr(anneal_train, "dispatch", forbidden)
    config = anneal_train.TrainingConfig(trial="181b", cluster="cw-rno2a")
    for operation in (anneal_validation.run, anneal_validation.dispatch):
        with pytest.raises(ValueError, match="bounded validation only"):
            operation(config)


def test_only_watchdog_suppressed_and_restored_on_failure(monkeypatch):
    original = faulthandler.dump_traceback_later
    cancellation = faulthandler.cancel_dump_traceback_later
    config = smoke()

    def native(received):
        assert received is config
        assert faulthandler.cancel_dump_traceback_later is cancellation
        assert faulthandler.dump_traceback_later is not original
        faulthandler.dump_traceback_later(120, repeat=True)
        raise RuntimeError("native validation failure")

    monkeypatch.setattr(anneal_validation, "_NATIVE_RUN", native)
    with pytest.raises(RuntimeError, match="native validation failure"):
        anneal_validation.run(config)
    assert faulthandler.dump_traceback_later is original


def test_dispatch_substitutes_worker_and_restores_on_failure(monkeypatch):
    original = anneal_train.run
    config = smoke()

    def native_dispatch(received):
        assert received is config
        assert anneal_train.run is anneal_validation.run
        raise RuntimeError("dispatch failure")

    monkeypatch.setattr(anneal_train, "dispatch", native_dispatch)
    with pytest.raises(RuntimeError, match="dispatch failure"):
        anneal_validation.dispatch(config)
    assert anneal_train.run is original


def test_equal_base_fim_loss_uses_targets_not_example_means():
    readouts = [
        (np.asarray([1.0], dtype=np.float32), np.asarray([0])),
        (np.asarray([3.0, 5.0, 7.0], dtype=np.float32), np.asarray([0, 0, 0])),
    ]
    assert scored_base_loss(readouts) == 4.0
    with pytest.raises(ValueError, match="scored targets"):
        scored_base_loss([])


def test_compact_metrics_keep_aggregate_charts_and_species_counts():
    raw = {"train/loss": 1.25, "moe/drop_fraction": 0.0, "moe/dropped_assignments": 0}
    for layer in range(8):
        raw[f"train/router/layer_{layer}/active_experts"] = 64 - layer
        raw[f"train/router/layer_{layer}/max_to_mean_load"] = 1.0 + layer / 10
        raw[f"train/router/layer_{layer}/routing_entropy"] = 2.0
    compact = anneal_train.compact_training_metrics(raw)
    assert compact["train/router/active_experts_min"] == 57
    assert compact["train/router/max_to_mean_load_max"] == pytest.approx(1.7)
    assert compact["train/loss"] == 1.25
    assert "moe/dropped_assignments" not in compact
    assert not any("/layer_" in key for key in compact)
    assert anneal_train.species_counts(
        {"species/Zmays": 12, "species/Athaliana": 5, "gap/1": 4}
    ) == {"Athaliana": 5, "Zmays": 12}
