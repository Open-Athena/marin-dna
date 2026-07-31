from __future__ import annotations

from pathlib import Path

import pytest

from marin_dna.pipelines.vertebrate_projection_dataset.provenance import (
    resolve_pipeline_commit,
)


def test_resolve_pipeline_commit_uses_valid_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", "A" * 40)
    assert resolve_pipeline_commit(tmp_path) == "a" * 40


def test_resolve_pipeline_commit_fails_without_git_or_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PIPELINE_COMMIT_SHA", raising=False)
    with pytest.raises(AssertionError, match="PIPELINE_COMMIT_SHA"):
        resolve_pipeline_commit(tmp_path)
