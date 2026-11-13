"""Sparse W&B-ready L2 norms of logical global parameters and reduced gradients."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormPolicy:
    """Collect scalars every 1000 updates, without per-expert histograms."""

    every_updates: int = 1000
    num_layers: int = 8

    def __post_init__(self) -> None:
        if self.every_updates <= 0 or self.num_layers <= 0:
            raise ValueError("Norm cadence and layer count must be positive")

    def due(self, update: int) -> bool:
        """Gradients are available only after a positive training update."""
        return update > 0 and update % self.every_updates == 0


def named_model_arrays(model: Any, tree_util: Any) -> dict[str, Any]:
    """Flatten a Grug/Equinox tree using jax.tree_util, without host transfers."""
    pairs, _ = tree_util.tree_flatten_with_path(model)
    return {
        tree_util.keystr(path): value
        for path, value in pairs
        if hasattr(value, "dtype")
        and (np.issubdtype(value.dtype, np.inexact) or str(value.dtype) == "bfloat16")
    }


def collect_norms(
    parameters: Mapping[str, Any],
    gradients: Mapping[str, Any],
    *,
    policy: NormPolicy | None = None,
    xp: Any = np,
) -> dict[str, Any]:
    """Compute on global arrays, never sum physical replicas or optimizer slots.

    Pass FP32 master parameters and globally reduced, unscaled gradients BEFORE
    clipping/optimizer transforms. For JAX pass xp=jax.numpy inside the compiled
    diagnostic branch. The returned scalars can join the ordinary W&B log dict.
    No array is copied to the host here. Overall norms include every float leaf
    once; selected projection norms cover all layers and the whole expert bank.
    """
    policy = policy or NormPolicy()
    if not parameters or parameters.keys() != gradients.keys():
        raise ValueError(
            "Parameters and gradients must have identical nonempty leaf sets"
        )
    for name in parameters:
        if parameters[name].shape != gradients[name].shape:
            raise ValueError(f"Parameter/gradient shape mismatch: {name}")
    selected = {
        "attention_q": ".attn.w_q",
        "router": ".mlp.router",
        "expert_down": ".mlp.expert_mlp.w_down",
    }
    output = {}
    for category, arrays in [("param", parameters), ("grad", gradients)]:
        totals = []
        for name, array in arrays.items():
            values = xp.asarray(array, dtype=xp.float32)
            totals.append(xp.sum(xp.square(values), dtype=xp.float32))
            for label, suffix in selected.items():
                if name.endswith(suffix):
                    if values.shape[0] != policy.num_layers:
                        raise ValueError(
                            "Expected stacked layer axis for selected projection"
                        )
                    output[f"norms/{category}/{label}"] = xp.sqrt(
                        xp.sum(xp.square(values), dtype=xp.float32)
                    )
        output[f"norms/{category}/overall"] = xp.sqrt(xp.sum(xp.stack(totals)))
    return output


def model_norms(
    parameters: Any,
    gradients: Any,
    tree_util: Any,
    *,
    xp: Any,
    policy: NormPolicy | None = None,
) -> dict[str, Any]:
    """Adapter for live Grug parameter and gradient trees at the training hook."""
    return collect_norms(
        named_model_arrays(parameters, tree_util),
        named_model_arrays(gradients, tree_util),
        policy=policy,
        xp=xp,
    )
