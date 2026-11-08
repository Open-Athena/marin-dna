import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from exp593_dna.recovery import bind_and_discover, commit_and_prune, prepare_checkpoint
from exp593_dna.schedule import TRAINING_TOKENS, TokenClock
from exp593_dna.train import (
    DNA216B_METADATA_DIGEST,
    DNA216B_RUN_ID,
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


def test_scientific_bindings_exclude_placement_and_checkpoint_policy():
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
    second = TrainingConfig(
        "cw-rno2a",
        nodes=8,
        checkpoint_interval_seconds=120,
        keep_temporary_checkpoints=30,
    )
    assert first.run_id == second.run_id == PRODUCTION_RUN_ID
    assert scientific_bindings(first, inputs, encoder) == scientific_bindings(
        second, inputs, encoder
    )
    assert first.checkpoint_root == second.checkpoint_root
    with pytest.raises(ValueError, match="identity must match"):
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


def test_short_checkpoint_cadence_retains_preparation_sources_and_permanent_saves(
    tmp_path,
):
    root = str(tmp_path)
    bindings = {"source": "same-scientific-model", "tokens": TRAINING_TOKENS}
    bind_and_discover(root, bindings)
    permanent, clock = metadata(tmp_path, 1, bindings, temporary=False)
    commit_and_prune(root, permanent, clock, bindings, keep_temporary=4)
    for step in range(2, 8):
        path, clock = metadata(tmp_path, step, bindings)
        commit_and_prune(root, path, clock, bindings, keep_temporary=4)
    assert (tmp_path / "step-1/metadata.json").is_file()
    assert not (tmp_path / "step-2").exists()
    assert not (tmp_path / "step-3").exists()
    assert all(
        (tmp_path / f"step-{step}/metadata.json").is_file() for step in range(4, 8)
    )
    assert bind_and_discover(root, bindings) == str(tmp_path / "step-7")
    previous_pointer = (tmp_path / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="retention"):
        commit_and_prune(root, path, clock, bindings, keep_temporary=0)
    assert (tmp_path / "latest.json").read_bytes() == previous_pointer
    for changes in (
        {"checkpoint_interval_seconds": 0},
        {"checkpoint_interval_seconds": -1},
        {"keep_temporary_checkpoints": 0},
    ):
        with pytest.raises(ValueError, match="positive integers"):
            TrainingConfig("cw-rno2a", **changes)


def test_dna_initialization_has_separate_identity_source_and_half_peak_rates():
    names = (
        "validation/primary_validation_512.jsonl",
        "audit/validation_disjoint_512.json",
        "audit/fim_validation_pairs.jsonl",
    )
    inputs = SimpleNamespace(
        manifest={"files": {name: {"sha256": "a" * 64} for name in names}}
    )
    encoder = SimpleNamespace(digest="b" * 64)
    text = TrainingConfig("cw-rno2a", nodes=4)
    dna = replace(text, initialization="dna216b", run_id=DNA216B_RUN_ID)
    original = scientific_bindings(text, inputs, encoder)
    adapted = scientific_bindings(dna, inputs, encoder)
    assert dna.checkpoint_root != text.checkpoint_root
    assert dna.source_checkpoint.endswith(
        "/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1/2026.10.08/checkpoints/step-412290"
    )
    assert adapted["source_metadata"] == DNA216B_METADATA_DIGEST
    assert adapted["source_checkpoint"] == dna.source_checkpoint
    assert adapted["run_id"] == DNA216B_RUN_ID
    expected_optimizer = {
        **original["optimizer"],
        "learning_rate": original["optimizer"]["learning_rate"] / 2,
        "adam_lr": original["optimizer"]["adam_lr"] / 2,
    }
    assert adapted == {
        **original,
        "run_id": DNA216B_RUN_ID,
        "source_checkpoint": dna.source_checkpoint,
        "source_metadata": DNA216B_METADATA_DIGEST,
        "optimizer": expected_optimizer,
    }
    assert adapted == scientific_bindings(replace(dna, nodes=8), inputs, encoder)
    for changes in (
        {"initialization": "dna216b"},
        {"run_id": DNA216B_RUN_ID},
        {"initialization": "unknown"},
    ):
        with pytest.raises(ValueError):
            replace(text, **changes)


def test_milestones_use_actual_input_counts():
    quarter = TRAINING_TOKENS // 4
    assert permanent_events(quarter - 1, quarter + 150000) == ("quarter_1",)
    assert permanent_events(TRAINING_TOKENS - 1, TRAINING_TOKENS + 150000) == ("final",)
