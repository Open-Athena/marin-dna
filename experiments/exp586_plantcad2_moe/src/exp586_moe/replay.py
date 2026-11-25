"""Compare live-copy execution variation with checkpoint-restored execution."""

import gc
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
from levanter.data.text.examples import GrugLmExample

from exp586_moe.schedule import TokenClock
from exp586_moe.state import (
    restore_dna_checkpoint,
    state_all_finite,
    state_error_by_group,
    state_max_abs_error,
)
from experiments.grug.moe_hero_ep.train import GrugTrainState


def independent_copy(state: GrugTrainState) -> GrugTrainState:
    """Copy buffers without aliasing while retaining memory kind and layout."""
    return jax.tree.map(
        lambda value: jax.device_put(value, value.format, may_alias=False), state
    )


def state_layout_equal(a: GrugTrainState, b: GrugTrainState) -> bool:
    return all(
        x.format == y.format and x.weak_type == y.weak_type
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def three_way_replay(
    state: GrugTrainState,
    clock: TokenClock,
    checkpoint: str,
    mesh: jax.sharding.Mesh,
    batches: Sequence[GrugLmExample],
    train_step: Callable[..., Any],
    log: Callable[[dict[str, float | int | bool | str]], None],
    *,
    tokenizer_sha256: str | None = None,
) -> dict[str, float | int | bool | str]:
    """Compare three branches while retaining at most two full states.

    Verify restoration before updating. Compare two independent live copies after
    each of two updates, release the second copy, then restore again and compare
    its two-update endpoint with the retained live endpoint. The six diagnostic
    updates never advance the primary checkpoint's clock. Differences are measured
    without approving a numerical tolerance. The checkpoint must have one writer.
    """
    if len(batches) != 2:
        raise ValueError("Replay control requires exactly two fixed batches")
    result: dict[str, float | int | bool | str] = {}

    def check_initial(name: str, other: GrugTrainState) -> None:
        error = float(state_max_abs_error(state, other))
        result[f"replay/{name}/initial_max_abs"] = error
        result[f"replay/{name}/layout_equal"] = state_layout_equal(state, other)
        if error != 0:
            raise ValueError(f"{name} differs before replay: {error}")
        for x, y in zip(jax.tree.leaves(state), jax.tree.leaves(other), strict=True):
            if x.dtype != y.dtype or x.sharding != y.sharding:
                raise ValueError(f"{name} changes dtype or sharding")

    def advance(
        current: GrugTrainState, batch: GrugLmExample, update: int
    ) -> tuple[GrugTrainState, dict]:
        next_state, metrics, _ = train_step(current, batch)
        jax.block_until_ready(next_state)
        if int(next_state.step) != clock.updates + update:
            raise ValueError("Replay update count differs")
        if not bool(state_all_finite(next_state)):
            raise ValueError("Nonfinite replay state")
        return next_state, metrics

    def compare(
        pair: str,
        update: int,
        left: GrugTrainState,
        right: GrugTrainState,
        left_metrics: dict,
        right_metrics: dict,
    ) -> None:
        prefix = f"replay/update_{update}/{pair}"
        measured: dict[str, float | int | bool | str] = {
            f"{prefix}/{key}": float(value)
            for key, value in state_error_by_group(left, right).items()
        }
        for key in (
            "train/loss",
            "train/router/routing_counts_per_layer",
            "qb_beta_per_layer",
        ):
            measured[f"{prefix}/metrics/{key}/max_abs"] = float(
                np.max(
                    np.abs(
                        np.asarray(left_metrics[key], dtype=np.float64)
                        - np.asarray(right_metrics[key], dtype=np.float64)
                    )
                )
            )
        result.update(measured)
        log(measured)

    log({"replay/stage": "initial_restore", "replay/max_live_states": 2})
    restored, restored_clock = restore_dna_checkpoint(
        state, checkpoint, mesh, data_seed=0, tokenizer_sha256=tokenizer_sha256
    )
    if clock != restored_clock:
        raise ValueError("Restored clock differs before replay")
    check_initial("restored", restored)
    del restored
    gc.collect()
    live_copy = independent_copy(state)
    check_initial("live_copy", live_copy)
    log(result)
    log({"replay/stage": "live_pair"})
    for update, batch in enumerate(batches, 1):
        state, metrics_a = advance(state, batch, update)
        live_copy, metrics_b = advance(live_copy, batch, update)
        compare("a_b", update, state, live_copy, metrics_a, metrics_b)
    del live_copy, metrics_b
    gc.collect()
    log({"replay/stage": "restored_branch"})
    restored, restored_clock = restore_dna_checkpoint(
        state, checkpoint, mesh, data_seed=0, tokenizer_sha256=tokenizer_sha256
    )
    if clock != restored_clock:
        raise ValueError("Restored clock changed between reads")
    for update, batch in enumerate(batches, 1):
        restored, metrics_r = advance(restored, batch, update)
    compare("a_r", 2, state, restored, metrics_a, metrics_r)
    result.update(
        {
            "replay/extra_updates": 6,
            "replay/max_live_states": 2,
            "replay/stage": "complete",
            "replay/restored_comparison_update": 2,
        }
    )
    return result
