"""Producer identity for immutable workflow result namespaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def resolve_pipeline_commit(repo_root: str | Path = ".") -> str:
    """Resolve and validate the producing commit from Sky or the local checkout."""
    override = os.environ.get("PIPELINE_COMMIT_SHA")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    git_head = result.stdout.strip() if result.returncode == 0 else None
    if override:
        commit = override.lower()
        assert COMMIT_PATTERN.fullmatch(commit), "invalid PIPELINE_COMMIT_SHA"
        assert git_head in {None, commit}, (
            f"PIPELINE_COMMIT_SHA {commit} does not match checkout HEAD {git_head}"
        )
        return commit
    assert git_head is not None and COMMIT_PATTERN.fullmatch(git_head), (
        "cannot resolve producing commit; set PIPELINE_COMMIT_SHA when .git is absent"
    )
    return git_head


def configuration_sha256(configuration: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible resolved Snakemake configuration."""
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
