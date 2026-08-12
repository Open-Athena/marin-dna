"""Producing-commit resolution for local and SkyPilot execution."""

from __future__ import annotations

import hashlib
import json
import os
import string
import subprocess
from collections.abc import Mapping
from pathlib import Path


def _validate_commit_sha(sha: str) -> str:
    assert len(sha) == 40 and set(sha) <= set(string.hexdigits), (
        f"expected a 40-character hexadecimal commit SHA, got {sha!r}"
    )
    return sha.lower()


def hash_pipeline_config(config: Mapping[str, object]) -> str:
    """Return a stable identity for the fully resolved Snakemake config."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_head(repo_root: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        _validate_commit_sha(result.stdout.strip()) if result.returncode == 0 else None
    )


def _producer_identity(
    *, pipeline_commit: str, config_sha256: str, pipeline_version: str, tier: str
) -> dict[str, str]:
    commit = _validate_commit_sha(pipeline_commit)
    assert len(config_sha256) == 64 and set(config_sha256) <= set(string.hexdigits)
    assert pipeline_version and tier in {"smoke", "full"}
    return {
        "pipeline_commit": commit,
        "config_sha256": config_sha256.lower(),
        "pipeline_version": pipeline_version,
        "tier": tier,
    }


def write_producer_manifest(
    path: str | Path,
    *,
    pipeline_commit: str,
    config_sha256: str,
    pipeline_version: str,
    tier: str,
) -> dict[str, str]:
    """Write the immutable identity attached to one result namespace."""
    identity = _producer_identity(
        pipeline_commit=pipeline_commit,
        config_sha256=config_sha256,
        pipeline_version=pipeline_version,
        tier=tier,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return identity


def validate_producer_manifest(path: str | Path, **expected: str) -> dict[str, str]:
    """Require a restored namespace to match its expected producer identity."""
    identity = json.loads(Path(path).read_text())
    assert identity == _producer_identity(**expected)
    return identity


def resolve_pipeline_commit(repo_root: str | Path = ".") -> str:
    """Resolve the producing SHA from an explicit env override or local Git.

    SkyPilot workdir sync intentionally omits ``.git``. Launches that generate
    dataset cards must therefore pass ``PIPELINE_COMMIT_SHA`` explicitly.
    """
    override = os.environ.get("PIPELINE_COMMIT_SHA")
    git_head = _git_head(repo_root)
    if override:
        commit = _validate_commit_sha(override)
        assert git_head in {None, commit}, (
            f"PIPELINE_COMMIT_SHA {commit} does not match checkout HEAD {git_head}"
        )
        return commit
    assert git_head is not None, (
        "cannot resolve producing commit; set PIPELINE_COMMIT_SHA when .git is absent"
    )
    return git_head
