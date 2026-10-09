import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.tensorstore_serialization import TensorStoreWriteConfig

from exp586_moe.config import PEAK_CHECKPOINT_UPDATE, TOTAL_UPDATES, TRAINING_TOKENS
from exp586_moe.schedule import TokenClock
from exp586_moe.train import (
    BATCH_SIZE,
    PERMANENT_CHECKPOINT_COUNT,
    PERMANENT_CHECKPOINT_UPDATES,
    TOKENS_PER_UPDATE,
    TrainingConfig,
    bind_and_discover,
    check_clock,
    checkpoint_due,
    commit_and_prune,
    config_json,
    is_permanent_checkpoint,
    prepare_checkpoint,
    primary_json,
    scientific_config,
    theoretical_flops,
    throughput_flop_metrics,
)
from experiments.grug.moe_hero_ep.train import _compute_flops


def test_trial_catalog_and_placement_do_not_change_science():
    a = TrainingConfig("pretrained", 1.0, "cw-rno2a", 2)
    b = dataclasses.replace(
        a, cluster="cw-us-east-02a", nodes=1, checkpoint_interval_seconds=300
    )
    assert a.run_id == b.run_id
    assert a.checkpoint_root == b.checkpoint_root
    assert scientific_config(a, "digest") == scientific_config(b, "digest")
    assert a.context_axis_size == b.context_axis_size == 1
    assert a.model.capacity_factor == 32
    assert a.model.pooled_transport_capacity_factor == 8
    scratch_ring = TrainingConfig("scratch", 1.0, "cw-rno2a", 2)
    scratch_pooled = dataclasses.replace(scratch_ring, moe_backend="pooled")
    assert scratch_ring.run_id == scratch_pooled.run_id
    assert scratch_ring.checkpoint_root == scratch_pooled.checkpoint_root
    assert scratch_ring.model.moe_implementation == "ring"
    assert scratch_ring.resolved_moe_backend == "ring"
    assert scratch_ring.model.pooled_transport_capacity_factor is None
    assert (
        scratch_ring.scientific_model.moe_implementation
        == "fixed_pooled_wave_all_to_all"
    )
    assert scientific_config(scratch_ring, "digest") == scientific_config(
        scratch_pooled, "digest"
    )
    assert TOTAL_UPDATES == 412_290
    assert a.schedule.peak_update == PEAK_CHECKPOINT_UPDATE == 4_123
    ids = {
        TrainingConfig(condition, multiplier, "cw-rno2a", 2).run_id
        for condition, multipliers in (
            ("pretrained", (0.5, 1.0, 2.0)),
            ("scratch", (1.0, 3.0, 10.0)),
        )
        for multiplier in multipliers
    }
    assert len(ids) == 6
    with pytest.raises(ValueError):
        TrainingConfig("scratch", 0.5, "cw-rno2a", 2)
    with pytest.raises(ValueError):
        dataclasses.replace(a, seed=1)
    with pytest.raises(ValueError):
        dataclasses.replace(a, nodes=16)
    with pytest.raises(ValueError, match="interval"):
        dataclasses.replace(a, checkpoint_interval_seconds=0)
    with pytest.raises(ValueError, match="backend"):
        dataclasses.replace(a, moe_backend="ring")


def test_recovery_checkpoint_after_resuming_without_waiting_full_interval():
    assert not checkpoint_due(368, 344, 80, 300)
    assert checkpoint_due(369, 344, 83, 300)
    assert checkpoint_due(460, 344, 300, 300)
    assert not checkpoint_due(460, 344, 300, 900)
    assert checkpoint_due(25, 0, 80, 900)
    assert checkpoint_due(PEAK_CHECKPOINT_UPDATE, 344, 1, 900)
    assert checkpoint_due(TOTAL_UPDATES, 344, 1, 900)


def test_eight_permanent_checkpoints_span_the_training_horizon():
    regular = sorted(
        PERMANENT_CHECKPOINT_UPDATES - {PEAK_CHECKPOINT_UPDATE, TOTAL_UPDATES}
    )
    expected_boundaries = [
        TRAINING_TOKENS * checkpoint_index / (PERMANENT_CHECKPOINT_COUNT - 1)
        for checkpoint_index in range(1, PERMANENT_CHECKPOINT_COUNT - 1)
    ]
    assert len(PERMANENT_CHECKPOINT_UPDATES) == PERMANENT_CHECKPOINT_COUNT
    assert len(regular) == len(expected_boundaries) == 6
    for update, boundary in zip(regular, expected_boundaries, strict=True):
        assert (update - 1) * TOKENS_PER_UPDATE < boundary
        assert update * TOKENS_PER_UPDATE >= boundary
        assert is_permanent_checkpoint(update)
        assert checkpoint_due(update, 0, 0, 900)
    assert is_permanent_checkpoint(PEAK_CHECKPOINT_UPDATE)
    assert is_permanent_checkpoint(TOTAL_UPDATES)
    assert not is_permanent_checkpoint(regular[0] - 1)


def test_complete_clock_contract():
    clock = TokenClock(TOKENS_PER_UPDATE, BATCH_SIZE * 8191, BATCH_SIZE, 1)
    check_clock(clock)
    for field in ("input_tokens", "loss_targets", "examples", "updates"):
        with pytest.raises(ValueError, match="cursor"):
            check_clock(
                dataclasses.replace(clock, **{field: getattr(clock, field) + 1})
            )


def test_mfu_and_total_gflops_match_levanter_convention_and_resume_clock():
    expected = {
        "pretrained": (1_220_124_672.0, 9_995_261_313_024.0),
        "scratch": (629_157_888.0, 5_154_061_418_496.0),
    }
    per_device, peak_16 = theoretical_flops("NVIDIA H100 80GB HBM3", 16)
    assert per_device == 989_500_000_000_000.0
    assert theoretical_flops("NVIDIA H100 80GB HBM3", 64) == (
        per_device,
        4 * peak_16,
    )

    for condition, (training_flops_per_token, expected_per_example) in expected.items():
        config = TrainingConfig(condition, 1.0, "cw-rno2a", 2)
        flops_per_example, summary = _compute_flops(model_config=config.model)
        assert flops_per_example == expected_per_example
        assert (
            3 * summary["throughput/flops_per_token_analytic"]
            == training_flops_per_token
        )

        completed_examples = 25 * BATCH_SIZE
        metrics = throughput_flop_metrics(
            flops_per_example=flops_per_example,
            completed_examples=completed_examples,
            batch_size=BATCH_SIZE,
            elapsed_seconds=2.0,
            theoretical_flops_total=peak_16,
        )
        assert metrics["throughput/mfu"] == pytest.approx(
            flops_per_example * BATCH_SIZE / 2.0 / peak_16 * 100.0
        )
        assert metrics["throughput/total_gflops"] == pytest.approx(
            flops_per_example * completed_examples / 1e9
        )

        resumed = throughput_flop_metrics(
            flops_per_example=flops_per_example,
            completed_examples=completed_examples + BATCH_SIZE,
            batch_size=BATCH_SIZE,
            elapsed_seconds=0.5,
            theoretical_flops_total=4 * peak_16,
        )
        assert resumed["throughput/mfu"] == pytest.approx(metrics["throughput/mfu"])
        assert resumed["throughput/total_gflops"] - metrics[
            "throughput/total_gflops"
        ] == pytest.approx(flops_per_example * BATCH_SIZE / 1e9)

        final = throughput_flop_metrics(
            flops_per_example=flops_per_example,
            completed_examples=TOTAL_UPDATES * BATCH_SIZE,
            batch_size=BATCH_SIZE,
            elapsed_seconds=2.0,
            theoretical_flops_total=peak_16,
        )
        assert final["throughput/total_gflops"] == pytest.approx(
            TRAINING_TOKENS * training_flops_per_token / 1e9
        )


def _metadata(step: int, digest: str, *, temporary: bool) -> dict:
    clock = TokenClock(
        step * TOKENS_PER_UPDATE,
        step * BATCH_SIZE * 8191,
        step * BATCH_SIZE,
        step,
    )
    return {
        "step": step,
        "timestamp": "2026-10-08T00:00:00+00:00",
        "is_temporary": temporary,
        "scientific_config_sha256": digest,
        "exp586_clock": dataclasses.asdict(clock),
    }


def test_checkpoint_binding_peak_and_bounded_retention(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setattr(
        TrainingConfig,
        "checkpoint_root",
        property(lambda self: str(root)),
    )
    config = TrainingConfig("scratch", 1.0, "cw-rno2a", 2)
    binding = scientific_config(config, "tokenizer")
    digest = hashlib.sha256(config_json(binding).encode()).hexdigest()
    saved_digest, latest = bind_and_discover(config, binding)
    assert saved_digest == digest
    assert latest is None
    assert (root / "training.json").read_text() == config_json(binding)
    with pytest.raises(ValueError, match="configuration changed"):
        bind_and_discover(config, {**binding, "changed": True})

    for step in range(1, 6):
        path = root / f"step-{step}"
        path.mkdir()
        (path / "metadata.json").write_text(
            json.dumps(_metadata(step, digest, temporary=step != 2))
        )
    (root / "step-6").mkdir()
    (root / "step-6" / "partial-array").write_text("incomplete")
    assert bind_and_discover(config, binding)[1] == str(root / "step-5")
    clock = TokenClock(
        5 * TOKENS_PER_UPDATE,
        5 * BATCH_SIZE * 8191,
        5 * BATCH_SIZE,
        5,
    )
    commit_and_prune(config, str(root / "step-5"), digest, clock)
    assert sorted(p.name for p in root.iterdir() if p.name.startswith("step-")) == [
        "step-2",
        "step-4",
        "step-5",
        "step-6",
    ]
    assert json.loads((root / "latest.json").read_text())["checkpoint"] == str(
        root / "step-5"
    )

    final = root / f"step-{TOTAL_UPDATES}"
    final.mkdir()
    (final / "metadata.json").write_text(
        json.dumps(_metadata(TOTAL_UPDATES, digest, temporary=False))
    )
    with pytest.raises(ValueError, match="Permanent peak checkpoint is missing"):
        bind_and_discover(config, binding)
    peak = root / f"step-{PEAK_CHECKPOINT_UPDATE}"
    peak.mkdir()
    (peak / "metadata.json").write_text(
        json.dumps(_metadata(PEAK_CHECKPOINT_UPDATE, digest, temporary=True))
    )
    with pytest.raises(ValueError, match="Permanent peak checkpoint is missing"):
        bind_and_discover(config, binding)
    (peak / "metadata.json").write_text(
        json.dumps(_metadata(PEAK_CHECKPOINT_UPDATE, digest, temporary=False))
    )
    assert bind_and_discover(config, binding)[1] == str(final)


def test_primary_io_broadcasts_failure():
    assert primary_json(lambda: [1, "ok"]) == [1, "ok"]
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        primary_json(lambda: 1 / 0)


def test_retry_clears_partial_layout_and_preserves_committed_saves(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        TrainingConfig, "checkpoint_root", property(lambda self: str(tmp_path))
    )
    config = TrainingConfig("scratch", 1.0, "cw-rno2a", 2)
    clock = TokenClock(TOKENS_PER_UPDATE, BATCH_SIZE * 8191, BATCH_SIZE, 1)
    path = primary_json(lambda: prepare_checkpoint(config, clock))
    with jax.set_mesh(compact_grug_mesh()):
        old = {"weights": jnp.zeros((4, 4))}
        new = {"weights": jnp.arange(16, dtype=jnp.float32).reshape(4, 4)}
        save_checkpoint(
            old, 1, path, write_config=TensorStoreWriteConfig(max_chunk_bytes=64)
        )
        (Path(path) / "metadata.json").unlink()
        with pytest.raises(ValueError, match="chunk_shape"):
            save_checkpoint(
                new, 1, path, write_config=TensorStoreWriteConfig(max_chunk_bytes=32)
            )
        sibling = Path(config.checkpoint_root) / "step-0"
        sibling.mkdir()
        (sibling / "metadata.json").write_text("committed checkpoint marker")
        assert primary_json(lambda: prepare_checkpoint(config, clock)) == path
        assert not Path(path).exists()
        assert (sibling / "metadata.json").read_text() == "committed checkpoint marker"
        save_checkpoint(
            new, 1, path, write_config=TensorStoreWriteConfig(max_chunk_bytes=32)
        )
        restored = load_checkpoint(new, path, mesh=jax.sharding.get_mesh())
        np.testing.assert_array_equal(restored["weights"], new["weights"])
        marker = (Path(path) / "metadata.json").read_bytes()
        with pytest.raises(RuntimeError, match="Refusing to overwrite"):
            primary_json(lambda: prepare_checkpoint(config, clock))
        assert (Path(path) / "metadata.json").read_bytes() == marker
