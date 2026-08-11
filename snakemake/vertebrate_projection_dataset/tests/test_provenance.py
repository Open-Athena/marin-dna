from __future__ import annotations

from pathlib import Path

import pytest
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    validate_producer_manifest,
    write_producer_manifest,
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


def test_environment_override_must_match_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", "a" * 40)
    monkeypatch.setattr(
        "marin_dna_vertebrate_projection.provenance._git_head",
        lambda _repo_root: "b" * 40,
    )
    with pytest.raises(AssertionError, match="does not match checkout HEAD"):
        resolve_pipeline_commit(tmp_path)


def test_hash_pipeline_config_is_stable_and_sensitive() -> None:
    first = hash_pipeline_config({"z": [1, 2], "a": True})
    reordered = hash_pipeline_config({"a": True, "z": [1, 2]})
    changed = hash_pipeline_config({"a": False, "z": [1, 2]})
    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_producer_manifest_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    path = tmp_path / "producer.json"
    expected = {
        "pipeline_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "pipeline_version": "v1",
        "tier": "smoke",
    }
    assert write_producer_manifest(path, **expected) == expected
    assert validate_producer_manifest(path, **expected) == expected

    path.write_text(path.read_text().replace('"tier": "smoke"', '"tier": "full"'))
    with pytest.raises(AssertionError):
        validate_producer_manifest(path, **expected)
