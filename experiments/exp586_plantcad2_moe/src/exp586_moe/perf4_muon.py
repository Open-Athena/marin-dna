"""Share independent expert-optimizer matrix work only within a physical node."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS

from experiments.grug.moe_hero_ep.grugmuon_hero import (
    _newtonschulz_batched_syrk,
    _zeropower_via_newtonschulz_local,
)


def _iterate_normalized_syrk(
    x: jax.Array, steps: int, coefficient_type: str
) -> jax.Array:
    """The existing polynomial after its materialized BF16 normalization boundary."""
    syrk = import_module("levanter.grug._moe.quack_symmetric_cute").quack_symmetric_gemm
    coefficients = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = jnp.swapaxes(x, -1, -2)
    for index in range(steps):
        a, b, c = coefficients[index % len(coefficients)]
        first = syrk(x)
        second = b * first + c * syrk(first)
        x = a * x + jnp.matmul(second, x)
    return jnp.swapaxes(x, -1, -2) if transpose else x


def node_local_expert_ns(
    mesh: jax.sharding.Mesh,
    *,
    group_size: int = 8,
    normalize_before_split: bool = False,
    slice_before_cast: bool = False,
    cast_before_gather: bool = False,
) -> Callable[..., jax.Array]:
    """Split independent matrices and gather the existing routine's returned dtype."""
    if normalize_before_split and slice_before_cast:
        raise ValueError("Select one explicit normalization-boundary experiment")
    if any(size != 1 for name, size in mesh.shape.items() if name != "replica_dcn"):
        raise ValueError("Node-local optimizer work requires fully replicated DP")
    devices = mesh.devices.reshape(-1).tolist()
    if group_size <= 1 or len(devices) % group_size:
        raise ValueError("Optimizer groups must divide the full DP mesh")
    groups = tuple(
        tuple(range(start, start + group_size))
        for start in range(0, len(devices), group_size)
    )
    if any(
        len({devices[i].process_index // 8 for i in group}) != 1 for group in groups
    ):
        raise ValueError("An optimizer work group would cross physical GPU nodes")

    def transform(
        path: Any,
        x: jax.Array,
        steps: int,
        eps: float,
        coefficient_type: str,
        use_syrk: bool,
        target_sharding: Any = None,
    ) -> jax.Array:
        del path
        if x.ndim != 4 or x.shape[0] * x.shape[1] % group_size:
            raise ValueError("Expert matrix count must divide the optimizer group")
        if target_sharding is not None and not target_sharding.is_fully_replicated:
            raise ValueError("Optimizer work must return to replicated parameters")

        def local(stack: jax.Array) -> jax.Array:
            # Match the existing QuACK path's BF16 input cast before calling
            # its helper; the non-QuACK reference receives the original dtype.
            local_stack = (
                stack.astype(jnp.bfloat16)
                if use_syrk and not slice_before_cast
                else stack
            )
            flat = local_stack.reshape((-1, *stack.shape[-2:]))
            if normalize_before_split:
                if not use_syrk:
                    raise ValueError(
                        "Full-stack normalization requires the existing QuACK path"
                    )
                flat = jax.lax.optimization_barrier(
                    flat / (jnp.linalg.norm(flat, axis=(-2, -1), keepdims=True) + eps)
                )
            count = flat.shape[0] // group_size
            position = jax.lax.axis_index("replica_dcn") % group_size
            chunk = jax.lax.dynamic_slice_in_dim(flat, position * count, count)
            if use_syrk and slice_before_cast:
                chunk = chunk.astype(jnp.bfloat16)
            if normalize_before_split:
                result = _iterate_normalized_syrk(chunk, steps, coefficient_type)
            elif use_syrk:
                result = _newtonschulz_batched_syrk(chunk, steps, eps, coefficient_type)
            else:
                result = jax.vmap(
                    lambda matrix: _zeropower_via_newtonschulz_local(
                        matrix, steps, eps, coefficient_type
                    )
                )(chunk)
            # Preserve the returned dtype. An extra BF16 cast can change FP32
            # values retained by XLA fusion, even though the inner matmuls use BF16.
            gathered = jax.lax.all_gather(
                result.astype(stack.dtype) if cast_before_gather else result,
                "replica_dcn",
                axis=0,
                axis_index_groups=groups,
                tiled=True,
            )
            return gathered.reshape(stack.shape).astype(stack.dtype)

        with jax.named_scope("node_local_muon"):
            return jax.shard_map(
                local, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False
            )(jax.sharding.reshard(x, P()))

    return transform
