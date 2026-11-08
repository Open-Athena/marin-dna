import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from exp593_dna.recovery import canonical_json
from exp593_dna.schedule import TokenClock
from exp593_dna.standby import (
    checked_checkpoint,
    committed_promotion,
    publish_once,
    validate_promotion,
)
from exp593_dna.train import TrainingConfig, scientific_bindings


def test_standby_controls_do_not_change_science_or_accept_smoke_identity():
    base = TrainingConfig("cw-rno2a", nodes=4)
    warm = replace(
        base,
        nodes=8,
        standby_id="reno-n8-v1",
        standby_checkpoint=base.checkpoint_root + "/step-100",
        standby_from_job="/eczech/old-root",
    )
    names = (
        "validation/primary_validation_512.jsonl",
        "audit/validation_disjoint_512.json",
        "audit/fim_validation_pairs.jsonl",
    )
    inputs = SimpleNamespace(
        manifest={"files": {n: {"sha256": "a" * 64} for n in names}}
    )
    encoder = SimpleNamespace(digest="b" * 64)
    assert scientific_bindings(base, inputs, encoder) == scientific_bindings(
        warm, inputs, encoder
    )
    assert warm.checkpoint_root == base.checkpoint_root
    assert warm.control_root.endswith("/placements/reno-n8-v1")
    for changes in (
        {"standby_id": "../unsafe"},
        {"standby_checkpoint": None},
        {"standby_from_job": "/eczech/old-root/child"},
        {"run_id": "dna-exp593-smoke", "validation_updates": 16},
    ):
        with pytest.raises(ValueError):
            replace(warm, **changes)


def test_promotion_requires_terminal_writer_exact_ready_and_full_state_clock(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    bindings = {"run_id": "dna-exp593-production"}
    (root / "training.json").write_text(canonical_json(bindings))
    checkpoint = root / "step-100"
    checkpoint.mkdir()
    clock = TokenClock(
        input_tokens=13107200,
        dna_tokens=13107200,
        loss_targets=13100800,
        examples=6400,
        updates=100,
    )
    raw = json.dumps(
        {
            "step": 100,
            "exp593_schema": 1,
            "exp593_clock": asdict(clock),
            "exp593_bindings": bindings,
        }
    ).encode()
    (checkpoint / "metadata.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    args = {
        "standby_id": "reno-n8-v1",
        "ready_sha256": "b" * 64,
        "old_root": "/eczech/old-root",
        "checkpoint_root": str(root),
        "bindings": bindings,
        "source_clock": clock,
    }
    command = {
        "protocol": "exp593-standby-promotion-v1",
        "standby_id": "reno-n8-v1",
        "ready_sha256": "b" * 64,
        "old_root": "/eczech/old-root",
        "old_child": "/eczech/old-root/train-dna-exp593-production",
        "root_state": "JOB_STATE_KILLED",
        "child_state": "JOB_STATE_KILLED",
        "checkpoint": str(checkpoint),
        "metadata_sha256": digest,
        "clock": asdict(clock),
    }
    assert validate_promotion(command, **args) == (str(checkpoint), clock)
    for changes in (
        {"root_state": "JOB_STATE_RUNNING"},
        {"child_state": "JOB_STATE_RUNNING"},
        {"ready_sha256": "c" * 64},
        {"old_child": "/eczech/another"},
        {"metadata_sha256": "d" * 64},
        {"clock": asdict(replace(clock, examples=6464))},
    ):
        with pytest.raises(ValueError):
            validate_promotion({**command, **changes}, **args)
    with pytest.raises(ValueError, match="rewind"):
        validate_promotion(
            command, **{**args, "source_clock": replace(clock, updates=101)}
        )
    with pytest.raises(ValueError, match="outside"):
        checked_checkpoint(str(root), str(checkpoint) + "/../step-100", bindings)
    # Read-only validation must not create a binding file when it is absent.
    (root / "training.json").unlink()
    with pytest.raises(FileNotFoundError):
        checked_checkpoint(str(root), str(checkpoint), bindings)
    assert not (root / "training.json").exists()


def test_standby_evidence_is_append_only(tmp_path):
    path = str(tmp_path / "ready.json")
    digest = publish_once(path, {"ready": True})
    assert (
        digest == hashlib.sha256(canonical_json({"ready": True}).encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="overwrite"):
        publish_once(path, {"ready": False})


def test_promoted_retry_uses_immutable_evidence_without_original_checkpoint(tmp_path):
    args = {
        "standby_id": "reno-n8-v1",
        "old_root": "/eczech/old-root",
        "run_id": "dna-exp593-production",
    }
    assert not committed_promotion(str(tmp_path), **args)
    ready_sha = publish_once(str(tmp_path / "ready.json"), {"probes": "completed"})
    command = {
        "protocol": "exp593-standby-promotion-v1",
        "standby_id": "reno-n8-v1",
        "ready_sha256": ready_sha,
        "old_root": "/eczech/old-root",
        "old_child": "/eczech/old-root/train-dna-exp593-production",
        "root_state": "JOB_STATE_KILLED",
        "child_state": "JOB_STATE_KILLED",
        "checkpoint": "already-pruned-after-promotion",
    }
    (tmp_path / "promote.json").write_text(json.dumps(command))
    assert committed_promotion(str(tmp_path), **args)
    (tmp_path / "ready.json").write_text('{"probes":"different"}')
    with pytest.raises(ValueError, match="readiness"):
        committed_promotion(str(tmp_path), **args)
