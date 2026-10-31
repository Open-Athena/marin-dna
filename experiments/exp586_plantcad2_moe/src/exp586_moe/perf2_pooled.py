# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated padding optimization for the pinned pooled EP backend.

Dispatch, buffer limits, routing, drops, and combine are unchanged. The local
wave function below is copied from the installed pinned Levanter version;
its only change selects compute_trimmed with an extra fast_rows argument.
"""

import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Bool, Float, Int
from levanter.grug._moe.common import (
    CapacityDrops,
    MoeDispatchCounts,
    _assignment_validity,
    _scaled_capacity,
    padding_skipped_assignments,
)
from levanter.grug._moe.ep_common import _ranks_within_groups
from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import (
    _combine_pooled,
    _compute_pooled,
    _dispatch_pooled,
    _PooledDispatch,
    _PooledOutput,
)
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import _axis_names, _token_spec_from_x


def compute_trimmed(
    dispatch: _PooledDispatch,
    *,
    moe_w13_local,
    moe_w2_local,
    activation_fn,
    fast_rows: int | tuple[int, ...],
) -> _PooledOutput:
    """Skip only unused trailing rows, falling back for a larger expert load."""
    capacity = dispatch.compacted_x.shape[1]
    if isinstance(fast_rows, int) and (fast_rows <= 0 or fast_rows >= capacity):
        return _compute_pooled(
            dispatch,
            moe_w13_local=moe_w13_local,
            moe_w2_local=moe_w2_local,
            activation_fn=activation_fn,
        )
    max_row = jnp.max(
        jnp.where(
            dispatch.receiver_keep, dispatch.receiver_linear_indices % capacity, -1
        )
    )

    def compute_small(row_count):
        result = _compute_pooled(
            dispatch._replace(compacted_x=dispatch.compacted_x[:, :row_count]),
            moe_w13_local=moe_w13_local,
            moe_w2_local=moe_w2_local,
            activation_fn=activation_fn,
        )
        return result._replace(
            compacted_output=jnp.pad(
                result.compacted_output, ((0, 0), (0, capacity - row_count), (0, 0))
            )
        )

    def compute_full():
        return _compute_pooled(
            dispatch,
            moe_w13_local=moe_w13_local,
            moe_w2_local=moe_w2_local,
            activation_fn=activation_fn,
        )

    if isinstance(fast_rows, tuple):
        if tuple(sorted(set(fast_rows))) != fast_rows or any(n <= 0 for n in fast_rows):
            raise ValueError(
                "Expert row tiers must be positive and strictly increasing"
            )
        tiers = tuple(n for n in fast_rows if n < capacity)
        if not tiers:
            return compute_full()
        branch = jnp.sum(max_row >= jnp.asarray(tiers), dtype=jnp.int32)
        return jax.lax.switch(
            branch, tuple(partial(compute_small, n) for n in tiers) + (compute_full,)
        )
    return jax.lax.cond(
        max_row < fast_rows, partial(compute_small, fast_rows), compute_full
    )


def _trimmed_pooled_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    token_valid_local: Bool[Array, "Tlocal"],  # noqa: F821
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
    transport_capacity_factor: float,
    num_expert_waves: int,
    fast_rows: int | tuple[int, ...],
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
    """Stripe each destination pool over fixed waves and report drops at each transport stage."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts}"
        )
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")
    if transport_capacity_factor <= 0:
        raise ValueError("transport_capacity_factor must be positive")
    if num_expert_waves <= 0:
        raise ValueError(f"num_expert_waves must be positive, got {num_expert_waves}")
    if local_experts % num_expert_waves != 0:
        raise ValueError(
            f"local expert count={local_experts} must be divisible by num_expert_waves={num_expert_waves}"
        )

    tokens_per_shard, hidden_dim = x_local.shape
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    num_waves = num_expert_waves
    physical_pool_capacity = max(
        math.ceil(
            transport_capacity_factor
            * assignments_per_shard
            / (expert_shards * num_waves)
        ),
        1,
    )
    physical_receiver_capacity = max(
        math.ceil(
            capacity_factor * assignments_per_shard / (local_experts * num_waves)
        ),
        1,
    )

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    assignment_valid = _assignment_validity(
        token_valid_local, tokens=tokens_per_shard, topk=topk
    )
    valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
    safe_experts = jnp.where(assignment_valid, flat_experts, 0)
    local_expert_indices = (safe_experts % local_experts).astype(jnp.int32)
    destination_shards = (safe_experts // local_experts).astype(jnp.int32)
    logical_pool_capacity = _scaled_capacity(
        valid_assignments,
        capacity_factor=transport_capacity_factor,
        divisor=expert_shards * num_waves,
        maximum=physical_pool_capacity,
    )
    # One scalar psum sizes the receiver limit from the EP group's valid demand; it is
    # consumed only after each wave's all-to-all, so it does not gate the first transport.
    global_valid_assignments = jax.lax.psum(valid_assignments, "expert")
    logical_receiver_capacity = _scaled_capacity(
        global_valid_assignments,
        capacity_factor=capacity_factor,
        divisor=num_experts * num_waves,
        maximum=physical_receiver_capacity,
    )
    # The quotient keeps one logical capacity pool per destination. The remainder
    # stripes that pool over equal static waves.
    destination_ranks = _ranks_within_groups(
        destination_shards,
        num_groups=expert_shards,
        valid=assignment_valid,
    )
    assignment_waves = destination_ranks % num_waves
    pool_ranks = destination_ranks // num_waves
    sender_keep = assignment_valid & (pool_ranks < logical_pool_capacity)

    remat = partial(
        jax.checkpoint,
        prevent_cse=False,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    out_local = jnp.zeros((tokens_per_shard, hidden_dim), dtype=jnp.float32)
    receiver_dropped = jnp.array(0, dtype=jnp.int32)
    for wave_index in range(num_waves):
        wave_sender_keep = sender_keep & (assignment_waves == wave_index)
        dispatch = partial(
            _dispatch_pooled,
            x_local=x_local,
            local_expert_indices=local_expert_indices,
            destination_shards=destination_shards,
            pool_ranks=pool_ranks,
            sender_keep=wave_sender_keep,
            local_experts=local_experts,
            expert_shards=expert_shards,
            pool_capacity=physical_pool_capacity,
            receiver_capacity=physical_receiver_capacity,
            receiver_limit=logical_receiver_capacity,
            assignments_per_shard=assignments_per_shard,
            topk=topk,
        )
        compute = partial(
            compute_trimmed,
            fast_rows=fast_rows,
            moe_w13_local=moe_w13_local,
            moe_w2_local=moe_w2_local,
            activation_fn=activation_fn,
        )
        combine = partial(
            _combine_pooled,
            combine_weights_local=combine_weights_local,
            expert_shards=expert_shards,
            pool_capacity=physical_pool_capacity,
        )
        pooled_dispatch = dispatch()
        pooled_output = remat(compute)(pooled_dispatch)
        out_local = out_local + remat(combine)(pooled_output)
        receiver_dropped = receiver_dropped + pooled_dispatch.receiver_dropped

    sender_dropped = valid_assignments - jnp.sum(sender_keep, dtype=jnp.int32)
    dropped_by_stage_local = jnp.stack((sender_dropped, receiver_dropped))
    dropped_by_stage = jax.lax.psum(dropped_by_stage_local, token_sharding_axes)
    drops = CapacityDrops(
        sender_dropped=dropped_by_stage[0], receiver_dropped=dropped_by_stage[1]
    )
    return out_local.astype(x_local.dtype), drops


def trimmed_mlp(
    expert: MoEExpertMlp,
    x,
    selected_experts,
    combine_weights,
    *,
    token_valid,
    mesh,
    report_capacity_overflow: bool,
    fast_rows: int | tuple[int, ...],
):
    """Use the same EP layout and weights, with guarded smaller expert GEMMs."""
    if expert.implementation != "fixed_pooled_wave_all_to_all":
        raise ValueError("Padding profile requires the existing pooled backend")
    if int(mesh.shape.get("expert", 1)) < 2:
        raise ValueError("Padding profile requires expert parallelism")
    token_spec = _token_spec_from_x(x, mesh)
    token_axes = _axis_names(token_spec[0])
    w13 = jnp.concatenate((expert.w_gate, expert.w_up), axis=-1)
    local = partial(
        _trimmed_pooled_local,
        activation_fn=expert.activation.to_jax_fn(),
        num_experts=w13.shape[0],
        capacity_factor=expert.capacity_factor,
        token_sharding_axes=token_axes,
        transport_capacity_factor=expert.pooled_transport_capacity_factor,
        num_expert_waves=expert.num_expert_waves,
        fast_rows=fast_rows,
    )
    specs = (
        token_spec,
        token_spec,
        token_spec,
        token_spec,
        P("expert", None, None),
        P("expert", None, None),
    )
    arrays = (x, selected_experts, combine_weights, token_valid, w13, expert.w_down)
    arrays = tuple(reshard(a, p) for a, p in zip(arrays, specs, strict=True))
    out, drops = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=specs,
        out_specs=(token_spec, CapacityDrops(sender_dropped=P(), receiver_dropped=P())),
        check_vma=False,
    )(*arrays)
    if report_capacity_overflow:
        return out, MoeDispatchCounts(
            sender_dropped=drops.sender_dropped,
            receiver_dropped=drops.receiver_dropped,
            padding_skipped=padding_skipped_assignments(
                token_valid, topk=selected_experts.shape[1]
            ),
        )
    return out
