import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from exp593_dna.recovery import bind_and_discover, commit_and_prune, prepare_checkpoint
from exp593_dna.schedule import TRAINING_TOKENS, TokenClock
from exp593_dna.train import (
    PRODUCTION_RUN_ID,
    TrainingConfig,
    permanent_events,
    scientific_bindings,
)


def metadata(root, step, bindings, *, temporary=True):
    clock = TokenClock(
        input_tokens=131072 * step,
        dna_tokens=131072 * step,
        loss_targets=131008 * step,
        examples=64 * step,
        updates=step,
    )
    path = root / f"step-{step}"
    path.mkdir()
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "step": step,
                "timestamp": f"2026-11-07T00:{step:02d}:00+00:00",
                "is_temporary": temporary,
                "exp593_schema": 1,
                "exp593_clock": asdict(clock),
                "exp593_bindings": bindings,
            }
        )
    )
    return str(path), clock


def test_recovery_is_append_only_and_prunes_only_old_temporary_checkpoints(tmp_path):
    root = str(tmp_path)
    bindings = {"source": "original-text", "tokens": TRAINING_TOKENS}
    assert bind_and_discover(root, bindings) is None
    permanent, clock = metadata(tmp_path, 1, bindings, temporary=False)
    commit_and_prune(root, permanent, clock, bindings)
    for step in (2, 3, 4):
        path, clock = metadata(tmp_path, step, bindings)
        commit_and_prune(root, path, clock, bindings)
    assert (tmp_path / "step-1/metadata.json").is_file()
    assert not (tmp_path / "step-2").exists()
    assert (tmp_path / "step-3/metadata.json").is_file()
    assert bind_and_discover(root, bindings) == str(tmp_path / "step-4")
    pointer = json.loads((tmp_path / "latest.json").read_text())
    assert pointer["clock"] == asdict(clock)
    assert len(pointer["metadata_sha256"]) == 64
    with pytest.raises(ValueError, match="overwrite"):
        prepare_checkpoint(root, clock)
    with pytest.raises(ValueError, match="configuration changed"):
        bind_and_discover(root, {**bindings, "source": "DNA-checkpoint"})
    partial_clock = replace(clock, updates=5)
    (tmp_path / "step-5").mkdir()
    retry = prepare_checkpoint(root, partial_clock)
    assert retry.startswith(str(tmp_path / "step-5-retry-"))
    assert not __import__("pathlib").Path(retry).exists()
    assert (tmp_path / "step-5").is_dir()


def test_scientific_bindings_exclude_placement_and_reject_another_final_model():
    names = (
        "validation/primary_validation_512.jsonl",
        "audit/validation_disjoint_512.json",
        "audit/fim_validation_pairs.jsonl",
    )
    inputs = SimpleNamespace(
        manifest={
            "files": {name: {"sha256": str(i) * 64} for i, name in enumerate(names)}
        }
    )
    encoder = SimpleNamespace(digest="a" * 64)
    first = TrainingConfig("cw-us-east-02a", nodes=1)
    second = TrainingConfig("cw-rno2a", nodes=8)
    assert first.run_id == second.run_id == PRODUCTION_RUN_ID
    assert scientific_bindings(first, inputs, encoder) == scientific_bindings(
        second, inputs, encoder
    )
    assert first.checkpoint_root == second.checkpoint_root
    with pytest.raises(ValueError, match="one final model"):
        TrainingConfig("cw-us-east-02a", run_id="another-production-model")
    with pytest.raises(ValueError, match="smoke identity"):
        TrainingConfig("cw-us-east-02a", validation_updates=16)
    smoke = TrainingConfig(
        "cw-us-east-02a", run_id="dna-exp593-loop-smoke-v1", validation_updates=16
    )
    assert smoke.complete(TokenClock(updates=16))
    assert "/validation/" in smoke.checkpoint_root
    assert not first.complete(
        TokenClock(input_tokens=TRAINING_TOKENS - 1, dna_tokens=TRAINING_TOKENS - 1)
    )
    assert first.complete(
        TokenClock(input_tokens=TRAINING_TOKENS + 1, dna_tokens=TRAINING_TOKENS + 1)
    )


def test_milestones_use_actual_input_counts():
    quarter = TRAINING_TOKENS // 4
    assert permanent_events(quarter - 1, quarter + 150000) == ("quarter_1",)
    assert permanent_events(TRAINING_TOKENS - 1, TRAINING_TOKENS + 150000) == ("final",)
