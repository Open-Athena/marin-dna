from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp479_mntp.pilot import (
    arm_is_complete,
    assert_observed_budget_projection,
    latest_local_checkpoint,
    selected_batch_size,
)


def test_selected_batch_size_requires_passing_preflight(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "memory_and_throughput": {"selected": {"batch_size": 128}},
            }
        ),
        encoding="utf-8",
    )
    assert selected_batch_size(path) == 128
    path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not pass"):
        selected_batch_size(path)


def test_latest_checkpoint_and_completion_manifest(tmp_path: Path) -> None:
    output = tmp_path / "arm"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step-0100.ckpt").touch()
    (checkpoints / "step-0200.ckpt").touch()
    assert latest_local_checkpoint(output) == checkpoints / "step-0200.ckpt"
    assert not arm_is_complete(output)
    (output / "manifest.json").write_text('{"global_step": 1000}\n', encoding="utf-8")
    assert arm_is_complete(output)


def test_observed_budget_projection_uses_completed_arm_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "transferred_mntp" / "runtime.json"
    runtime.parent.mkdir()
    runtime.write_text('{"elapsed_seconds": 3600}\n', encoding="utf-8")
    monkeypatch.setenv("EXP479_INSTANCE_START_UNIX", "0")
    monkeypatch.setattr("exp479_mntp.pilot.time.time", lambda: 3600.0)
    projection = assert_observed_budget_projection(tmp_path)
    assert projection is not None
    assert projection["completed_arms"] == 1
    assert projection["remaining_arms"] == 2
    assert projection["projected_total_usd"] < 50
