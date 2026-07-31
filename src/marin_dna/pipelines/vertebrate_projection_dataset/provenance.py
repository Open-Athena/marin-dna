"""Producing-commit resolution for local and SkyPilot execution."""

from __future__ import annotations

import os
import string
import subprocess
from pathlib import Path


def _validate_commit_sha(sha: str) -> str:
    assert len(sha) == 40 and set(sha) <= set(string.hexdigits), (
        f"expected a 40-character hexadecimal commit SHA, got {sha!r}"
    )
    return sha.lower()


def resolve_pipeline_commit(repo_root: str | Path = ".") -> str:
    """Resolve the producing SHA from an explicit env override or local Git.

    SkyPilot workdir sync intentionally omits ``.git``. Launches that generate
    dataset cards must therefore pass ``PIPELINE_COMMIT_SHA`` explicitly.
    Bootstrap-only targets do not call this function.
    """
    override = os.environ.get("PIPELINE_COMMIT_SHA")
    if override:
        return _validate_commit_sha(override)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "cannot resolve producing commit; set PIPELINE_COMMIT_SHA when .git is absent"
    )
    return _validate_commit_sha(result.stdout.strip())
