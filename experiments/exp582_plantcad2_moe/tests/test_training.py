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

from exp582_moe.schedule import TokenClock
from exp582_moe.train import (
    BATCH_SIZE,
    COOLDOWN_UPDATE,
    TOKENS_PER_UPDATE,
    TOTAL_UPDATES,
    TrainingConfig,
    bind_and_discover,
    check_clock,
    checkpoint_due,
    commit_and_prune,
    config_json,
    legacy_scientific_config,
    prepare_checkpoint,
    primary_json,
    scientific_config,
    validate_transition,
)


def test_placement_does_not_change_trial_or_science():
    a = TrainingConfig("pretrained", 1.0, "cw-rno2a", 8)
    b = dataclasses.replace(
        a, cluster="cw-us-east-02a", nodes=4, checkpoint_interval_seconds=300
    )
    assert a.run_id == b.run_id
    assert a.checkpoint_root == b.checkpoint_root
    assert scientific_config(a, "digest") == scientific_config(b, "digest")
    expanded = dataclasses.replace(a, nodes=16)
    assert expanded.context_axis_size == 2
    assert a.context_axis_size == b.context_axis_size == 1
    assert scientific_config(a, "digest") == scientific_config(expanded, "digest")
    assert expanded.run_id == a.run_id
    assert a.model.capacity_factor == 32
    assert a.model.pooled_transport_capacity_factor == 8
    assert (TOTAL_UPDATES, COOLDOWN_UPDATE) == (21440, 10720)
    assert a.schedule.peak_update == 10720
    assert dataclasses.replace(a, lr_multiplier=0.5).schedule.peak_update == 10772
    assert (
        len(
            {
                TrainingConfig(c, m, "cw-rno2a", 8).run_id
                for c in ("pretrained", "random")
                for m in (0.5, 1.0, 2.0)
            }
        )
        == 6
    )
    with pytest.raises(ValueError):
        dataclasses.replace(a, seed=1)
    with pytest.raises(ValueError):
        dataclasses.replace(a, nodes=1)
    with pytest.raises(ValueError, match="interval"):
        dataclasses.replace(a, checkpoint_interval_seconds=0)


def test_recovery_checkpoint_after_resuming_without_waiting_full_interval():
    assert not checkpoint_due(368, 344, 80, 300)
    assert checkpoint_due(369, 344, 83, 300)
    assert not checkpoint_due(370, 344, 3, 300)
    assert checkpoint_due(460, 344, 300, 300)
    assert not checkpoint_due(460, 344, 300, 900)
    assert checkpoint_due(25, 0, 80, 900)
    assert checkpoint_due(COOLDOWN_UPDATE, 344, 1, 900)
    assert checkpoint_due(TOTAL_UPDATES, 344, 1, 900)
    assert checkpoint_due(10772, 10771, 1, 900, peak_update=10772)
    assert not checkpoint_due(10720, 10000, 1, 900, peak_update=10772)


def test_complete_clock_contract():
    clock = TokenClock(TOKENS_PER_UPDATE, BATCH_SIZE * 8191, BATCH_SIZE, 1)
    check_clock(clock)
    for field in ("input_tokens", "loss_targets", "examples", "updates"):
        with pytest.raises(ValueError, match="cursor"):
            check_clock(
                dataclasses.replace(clock, **{field: getattr(clock, field) + 1})
            )


def test_checkpoint_binding_and_bounded_retention(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setattr(
        TrainingConfig,
        "checkpoint_root",
        property(lambda self, original=root: str(original)),
    )
    config = TrainingConfig("random", 0.5, "cw-rno2a", 4)
    legacy = config_json(legacy_scientific_config(config, "tokenizer"))
    legacy_digest = hashlib.sha256(legacy.encode()).hexdigest()
    (tmp_path / "training.json").write_text(legacy)
    source = root / "step-1"
    source.mkdir()
    source_clock = TokenClock(TOKENS_PER_UPDATE, BATCH_SIZE * 8191, BATCH_SIZE, 1)
    source_metadata = {
        "step": 1,
        "timestamp": "2026-10-06T00:00:00+00:00",
        "exp582_schema": 1,
        "data_seed": 0,
        "is_temporary": True,
        "tokenizer_sha256": "tokenizer",
        "scientific_config_sha256": legacy_digest,
        "exp582_clock": dataclasses.asdict(source_clock),
    }
    source_blob = json.dumps(source_metadata)
    (source / "metadata.json").write_text(source_blob)
    transition = {
        "checkpoint": str(source),
        "metadata_sha256": hashlib.sha256(source_blob.encode()).hexdigest(),
        "scientific_config_sha256": legacy_digest,
        "clock": dataclasses.asdict(source_clock),
        "peak_update": COOLDOWN_UPDATE,
    }
    monkeypatch.setattr(TrainingConfig, "transition", property(lambda self: transition))
    binding = scientific_config(config, "tokenizer")
    digest, latest = bind_and_discover(config, binding)
    assert digest == hashlib.sha256(config_json(binding).encode()).hexdigest()
    assert latest == str(source)
    with pytest.raises(ValueError, match="configuration changed"):
        bind_and_discover(config, {**binding, "changed": True})
    with pytest.raises(ValueError, match="scientific configuration mismatch"):
        validate_transition(config, "wrong-tokenizer")
    (source / "metadata.json").write_text(source_blob + " ")
    with pytest.raises(ValueError, match="metadata digest mismatch"):
        validate_transition(config, "tokenizer")
    (source / "metadata.json").write_text(source_blob)
    old_root = root
    root = root / "short-linear-v2"
    for step in range(1, 6):
        path = root / f"step-{step}"
        path.mkdir()
        clock = TokenClock(
            step * TOKENS_PER_UPDATE, step * BATCH_SIZE * 8191, step * BATCH_SIZE, step
        )
        metadata = {
            "step": step,
            "timestamp": "2026-10-06T00:00:00+00:00",
            "is_temporary": step != 2,
            "scientific_config_sha256": digest,
            "exp582_clock": dataclasses.asdict(clock),
        }
        (path / "metadata.json").write_text(json.dumps(metadata))
    # An incomplete future save must not become the latest or be counted in retention.
    (root / "step-6").mkdir()
    (root / "step-6" / "partial-array").write_text("incomplete")
    assert bind_and_discover(config, binding)[1] == str(root / "step-5")
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
    assert (source / "metadata.json").read_text() == source_blob
    assert (old_root.parent / "training.json").read_text() == legacy
    with pytest.raises(ValueError, match="readback mismatch"):
        commit_and_prune(config, str(root / "step-5"), "wrong", clock)
    final = root / f"step-{TOTAL_UPDATES}"
    final.mkdir()
    final_metadata = {**metadata, "step": TOTAL_UPDATES, "is_temporary": False}
    (final / "metadata.json").write_text(json.dumps(final_metadata))
    with pytest.raises(ValueError, match="Permanent peak checkpoint is missing"):
        bind_and_discover(config, binding)
    peak = root / f"step-{COOLDOWN_UPDATE}"
    peak.mkdir()
    peak_metadata = {**metadata, "step": COOLDOWN_UPDATE, "is_temporary": True}
    (peak / "metadata.json").write_text(json.dumps(peak_metadata))
    with pytest.raises(ValueError, match="Permanent peak checkpoint is missing"):
        bind_and_discover(config, binding)
    peak_metadata["is_temporary"] = False
    (peak / "metadata.json").write_text(json.dumps(peak_metadata))
    assert bind_and_discover(config, binding)[1] == str(final)


def test_primary_io_broadcasts_failure():
    assert primary_json(lambda: [1, "ok"]) == [1, "ok"]
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        primary_json(lambda: 1 / 0)


def test_retry_clears_partial_chunk_layout_and_preserves_committed_saves(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        TrainingConfig, "checkpoint_root", property(lambda self: str(tmp_path))
    )
    config = TrainingConfig("random", 1.0, "cw-rno2a", 8)
    clock = TokenClock(TOKENS_PER_UPDATE, BATCH_SIZE * 8191, BATCH_SIZE, 1)
    path = primary_json(lambda: prepare_checkpoint(config, clock))
    with jax.set_mesh(compact_grug_mesh()):
        old = {"weights": jnp.zeros((4, 4))}
        new = {"weights": jnp.arange(16, dtype=jnp.float32).reshape(4, 4)}
        save_checkpoint(
            old, 1, path, write_config=TensorStoreWriteConfig(max_chunk_bytes=64)
        )
        # A failed save has array metadata but no checkpoint completion marker.
        (Path(path) / "metadata.json").unlink()
        with pytest.raises(ValueError, match="chunk_shape"):
            save_checkpoint(
                new, 1, path, write_config=TensorStoreWriteConfig(max_chunk_bytes=32)
            )
        assert not (Path(path) / "metadata.json").exists()
        sibling = Path(config.revision_root) / "step-0"
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
        restored = load_checkpoint(new, path, mesh=jax.sharding.get_mesh())
        np.testing.assert_array_equal(restored["weights"], new["weights"])
