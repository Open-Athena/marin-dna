"""Independent diagnostic reference for BF16 atomic scatter accumulation.

Keep the native grouped expert projections. Restore each token's top-k order,
weight and sum contributions in FP32, and round the result once.
This changes only a read-only reference, never training or benchmark scoring.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _prepare_moe_dispatch_indices_with_assignment_ids,
    _zero_dropped_assignments,
    _zero_inactive_grouped_rows,
    split_moe_w13_output,
)


def ordered_combine(
    dispatched: jax.Array, positions: jax.Array, weights: jax.Array
) -> jax.Array:
    """Undo expert sorting and accumulate each token's contributions in FP32."""
    contributions = dispatched[positions].astype(jnp.float32) * weights[
        ..., None
    ].astype(jnp.float32)
    return jnp.sum(contributions, axis=1).astype(dispatched.dtype)


def stable_scatter(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    token_valid: jax.Array,
    moe_w13: jax.Array,
    moe_w2: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[jax.Array, jax.Array]:
    tokens, positions, sizes, _ = _prepare_moe_dispatch_indices_with_assignment_ids(
        selected_experts, token_valid, num_experts=num_experts
    )
    cumulative = jnp.cumsum(sizes).astype(jnp.int32)
    inputs = _zero_inactive_grouped_rows(x[tokens], cumulative)
    projected = ragged_dot(inputs, moe_w13, sizes)
    gate, up = split_moe_w13_output(
        projected, intermediate_dim=moe_w2.shape[1], interleaved=False
    )
    dispatched = ragged_dot(activation_fn(gate) * up, moe_w2, sizes)
    dispatched = _zero_inactive_grouped_rows(dispatched, cumulative)
    weights = jnp.where(token_valid[:, None], combine_weights, 0)
    return ordered_combine(dispatched, positions, weights), _zero_dropped_assignments()


def install_stable_scatter_reference() -> None:
    """Replace only the local scatter dispatch after clearing compiled traces."""
    from levanter.grug._moe import local

    local._MOE_LOCAL_FNS["scatter"] = stable_scatter
