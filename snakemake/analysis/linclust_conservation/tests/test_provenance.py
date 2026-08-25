from __future__ import annotations

import subprocess

import pytest
from marin_dna_linclust_conservation.provenance import (
    configuration_sha256,
    resolve_pipeline_commit,
)


def test_configuration_sha256_is_order_independent() -> None:
    assert configuration_sha256({"b": 2, "a": {"x": 1}}) == configuration_sha256(
        {"a": {"x": 1}, "b": 2}
    )


def test_explicit_pipeline_commit_must_match_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", head)
    assert resolve_pipeline_commit() == head
    monkeypatch.setenv("PIPELINE_COMMIT_SHA", "0" * 40)
    with pytest.raises(AssertionError, match="does not match"):
        resolve_pipeline_commit()
