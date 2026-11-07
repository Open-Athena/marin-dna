"""Full optimizer checkpoints bound to variable-token and source-row clocks."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import jax
from experiments.grug.moe_hero_ep.train import GrugTrainState
from levanter.checkpoint import load_checkpoint, save_checkpoint
from rigging.filesystem.storage_path import StoragePath

from exp593_dna.schedule import TokenClock


def validate_checkpoint_metadata(
    metadata: dict[str, Any], bindings: dict[str, Any]
) -> TokenClock:
    """Reject changed data, tokenizer, objective or initialization before restore."""
    if (
        metadata.get("exp593_schema") != 1
        or metadata.get("exp593_bindings") != bindings
    ):
        raise ValueError("Augmented checkpoint scientific/source bindings differ")
    clock = TokenClock(**metadata["exp593_clock"])
    if metadata.get("step") != clock.updates:
        raise ValueError("Checkpoint step and token/data clock differ")
    return clock


def save_training_checkpoint(
    state: GrugTrainState,
    clock: TokenClock,
    path: str,
    bindings: dict[str, Any],
    *,
    temporary: bool = False,
) -> None:
    """Serialize global arrays, including FP32 masters, momentum and router state."""
    if int(state.step) != clock.updates or int(state.opt_state.count) != clock.updates:
        raise ValueError("Model, optimizer and data clocks differ")
    if state.master_params is None or state.ema_params is not None:
        raise ValueError("Checkpoint requires the approved FP32-master state")
    if (StoragePath(path) / "metadata.json").exists():
        raise ValueError("Refusing to overwrite a committed checkpoint")
    save_checkpoint(
        state,
        clock.updates,
        path,
        is_temporary=temporary,
        metadata={
            "exp593_schema": 1,
            "exp593_clock": asdict(clock),
            "exp593_bindings": bindings,
        },
    )


def restore_training_checkpoint(
    exemplar: GrugTrainState,
    path: str,
    mesh: jax.sharding.Mesh,
    bindings: dict[str, Any],
) -> tuple[GrugTrainState, TokenClock]:
    """Restore into current global-array shardings without deleting the exemplar."""
    metadata = json.loads((StoragePath(path) / "metadata.json").read_bytes())
    clock = validate_checkpoint_metadata(metadata, bindings)
    template = jax.tree.map(
        lambda value: (
            jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding)
            if isinstance(value, jax.Array)
            else value
        ),
        exemplar,
    )
    restored = load_checkpoint(template, path, mesh=mesh, allow_partial=False)
    if (
        int(restored.step) != clock.updates
        or int(restored.opt_state.count) != clock.updates
    ):
        raise ValueError("Restored optimizer/model clocks differ from metadata")
    if restored.master_params is None or restored.ema_params is not None:
        raise ValueError("Restored checkpoint omitted required FP32 training state")
    return restored, clock
