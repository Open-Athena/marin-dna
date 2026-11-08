"""Append-only checkpoints, strict source bindings and one latest recovery pointer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any
from uuid import uuid4

import fsspec
from levanter.checkpoint import discover_checkpoint_candidates
from rigging.filesystem.storage_path import StoragePath

from exp593_dna.schedule import TokenClock
from exp593_dna.state import validate_checkpoint_metadata


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bind_and_discover(root: str, bindings: dict[str, Any]) -> str | None:
    """Called by the primary only, before allocating the production model."""
    serialized = canonical_json(bindings)
    path = StoragePath(root) / "training.json"
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError("Trial scientific configuration changed")
    else:
        path.write_text(serialized)
        if path.read_text() != serialized:
            raise ValueError("Scientific configuration readback failed")
    candidates = discover_checkpoint_candidates(root)
    for candidate in candidates:
        clock = validate_checkpoint_metadata(candidate.metadata, bindings)
        if clock.examples != clock.updates * 64:
            raise ValueError("Recovery cursor is not a whole approved global batch")
    return candidates[-1].path if candidates else None


def prepare_checkpoint(root: str, clock: TokenClock) -> str:
    """Never reuse a partial TensorStore destination after a failed write."""
    path = f"{root}/step-{clock.updates}"
    fs, key = fsspec.core.url_to_fs(path)
    fs.invalidate_cache(key)
    if fs.exists(key + "/metadata.json"):
        raise ValueError("Refusing to overwrite a committed checkpoint")
    if fs.exists(key):
        path += "-retry-" + uuid4().hex
        retry_fs, retry_key = fsspec.core.url_to_fs(path)
        if retry_fs.exists(retry_key):
            raise RuntimeError("Fresh checkpoint destination already exists")
    return path


def commit_and_prune(
    root: str,
    checkpoint: str,
    clock: TokenClock,
    bindings: dict[str, Any],
    *,
    keep_temporary: int = 2,
) -> str:
    """Read back the commit and pointer before pruning older temporary saves."""
    if not isinstance(keep_temporary, int) or keep_temporary < 1:
        raise ValueError("Temporary checkpoint retention must be a positive integer")
    expected = f"{root}/step-{clock.updates}"
    if checkpoint != expected and not checkpoint.startswith(expected + "-retry-"):
        raise ValueError("Checkpoint is outside the active trial")
    raw_metadata = (StoragePath(checkpoint) / "metadata.json").read_bytes()
    metadata = json.loads(raw_metadata)
    if validate_checkpoint_metadata(metadata, bindings) != clock:
        raise ValueError("Checkpoint commit clock differs")
    digest = hashlib.sha256(raw_metadata).hexdigest()
    payload = canonical_json(
        {"checkpoint": checkpoint, "metadata_sha256": digest, "clock": asdict(clock)}
    )
    pointer = StoragePath(root) / "latest.json"
    pointer.write_text(payload)
    if pointer.read_text() != payload:
        raise ValueError("Latest recovery pointer readback failed")
    temporary = [
        c
        for c in discover_checkpoint_candidates(root)
        if c.metadata.get("is_temporary")
        and c.metadata.get("exp593_bindings") == bindings
    ]
    for candidate in temporary[:-keep_temporary]:
        if (
            not candidate.path.startswith(root + "/step-")
            or candidate.step >= clock.updates
        ):
            raise ValueError("Unsafe checkpoint retention candidate")
        fs, key = fsspec.core.url_to_fs(candidate.path)
        fs.rm(key, recursive=True)
    return digest
