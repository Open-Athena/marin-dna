"""Explicit, immutable promotion gate for an allocation with no production writer."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from exp593_dna.recovery import canonical_json
from exp593_dna.schedule import TokenClock
from exp593_dna.state import validate_checkpoint_metadata

TERMINAL_STATES = {"JOB_STATE_KILLED", "JOB_STATE_FAILED", "JOB_STATE_SUCCEEDED"}


def validate_standby_id(value: str) -> None:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", value) is None:
        raise ValueError("Invalid standby identity")


def checked_checkpoint(
    root: str, checkpoint: str, bindings: dict[str, Any], digest: str | None = None
) -> TokenClock:
    """Read existing production metadata without binding, pruning or writing it."""
    if not checkpoint.startswith(root + "/step-") or "/" in checkpoint[len(root) + 1 :]:
        raise ValueError("Standby checkpoint is outside the production run")
    if (StoragePath(root) / "training.json").read_text() != canonical_json(bindings):
        raise ValueError("Standby scientific bindings differ")
    raw = (StoragePath(checkpoint) / "metadata.json").read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Promotion checkpoint digest differs")
    clock = validate_checkpoint_metadata(json.loads(raw), bindings)
    if clock.examples != clock.updates * 64:
        raise ValueError("Standby checkpoint is not a whole global batch")
    return clock


def publish_once(path: str, payload: dict[str, Any]) -> str:
    """Publish only under a unique allocation's control/artifact namespace."""
    raw = canonical_json(payload)
    destination = StoragePath(path)
    if destination.exists():
        raise ValueError("Refusing to overwrite standby control or evidence")
    destination.write_text(raw)
    if destination.read_text() != raw:
        raise ValueError("Standby publication readback failed")
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_promotion_identity(
    command: dict[str, Any],
    *,
    standby_id: str,
    ready_sha256: str,
    old_root: str,
    bindings: dict[str, Any],
) -> None:
    """Require explicit operator evidence that both old writer jobs are terminal.

    The operator queries Iris immediately before publishing this command.
    Merely acquiring capacity, becoming ready, or seeing a checkpoint never
    promotes a standby and never grants it access to the production tracker.
    """
    expected = {
        "protocol": "exp593-standby-promotion-v1",
        "standby_id": standby_id,
        "ready_sha256": ready_sha256,
        "old_root": old_root,
        "old_child": old_root + "/train-" + bindings["run_id"],
    }
    if any(command.get(key) != value for key, value in expected.items()):
        raise ValueError("Promotion identity or readiness receipt differs")
    if any(
        command.get(key) not in TERMINAL_STATES for key in ("root_state", "child_state")
    ):
        raise ValueError("Old production writer is not terminal")


def validate_promotion(
    command: dict[str, Any],
    *,
    standby_id: str,
    ready_sha256: str,
    old_root: str,
    checkpoint_root: str,
    bindings: dict[str, Any],
    source_clock: TokenClock,
) -> tuple[str, TokenClock]:
    """Require the committed handoff checkpoint before the first promotion."""
    validate_promotion_identity(
        command,
        standby_id=standby_id,
        ready_sha256=ready_sha256,
        old_root=old_root,
        bindings=bindings,
    )
    checkpoint = command["checkpoint"]
    digest = command["metadata_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Promotion requires an exact checkpoint digest")
    clock = checked_checkpoint(checkpoint_root, checkpoint, bindings, digest)
    if (
        clock.input_tokens < source_clock.input_tokens
        or clock.updates < source_clock.updates
    ):
        raise ValueError("Promotion would rewind the standby source checkpoint")
    if command.get("clock") != asdict(clock):
        raise ValueError("Promotion clock differs from committed metadata")
    return checkpoint, clock


def committed_promotion(
    control_root: str, *, standby_id: str, old_root: str, run_id: str
) -> bool:
    """Recognize re-entry into an already promoted allocation after an Iris retry.

    The original handoff checkpoint can have been pruned by now.
    Its immutable promotion/readiness evidence still fences the old writer;
    normal strict production discovery chooses the newest recovery checkpoint.
    """
    path = StoragePath(control_root) / "promote.json"
    if not path.exists():
        return False
    ready = (StoragePath(control_root) / "ready.json").read_bytes()
    validate_promotion_identity(
        json.loads(path.read_text()),
        standby_id=standby_id,
        ready_sha256=hashlib.sha256(ready).hexdigest(),
        old_root=old_root,
        bindings={"run_id": run_id},
    )
    return True
