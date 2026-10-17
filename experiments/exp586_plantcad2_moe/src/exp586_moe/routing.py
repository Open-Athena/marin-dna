"""Diagnostic controls around the unchanged native expert-parallel step."""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax


def damp_qb_updates(train_step: Callable[..., Any], rate: float) -> Callable[..., Any]:
    """Interpolate pending global-histogram QB thresholds before the next update.

    The native model still centers and applies the pending thresholds normally.
    Interpolation lives inside the outer JIT so donation cannot destroy the old
    threshold before it is used. Rate one returns the unmodified native callable.
    """
    if not 0 < rate <= 1:
        raise ValueError("QB update rate must lie in (0, 1]")
    if rate == 1:
        return train_step

    @functools.partial(jax.jit, donate_argnums=(0,))
    def step(state: Any, batch: Any) -> Any:
        previous = state.pending_qb_betas
        updated, metrics, watch = train_step(state, batch)
        updated = dataclasses.replace(
            updated,
            pending_qb_betas=(1 - rate) * previous + rate * updated.pending_qb_betas,
        )
        return updated, metrics, watch

    return step
