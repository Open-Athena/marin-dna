"""Partition only independent expert optimizer arrays; retain a full model replica."""

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep.grugmuon_hero import (
    _newtonschulz_batched_syrk,
    _target_named_sharding,
    _zeropower_via_newtonschulz_local,
)


def _is_expert_array(path: Any, value: Any) -> bool:
    """Select only stacked independent expert matrices, never dense matrices."""
    return (
        isinstance(value, jax.Array)
        and value.ndim == 4
        and "expert_mlp" in jax.tree_util.keystr(path)
    )


def partition_expert_optimizer_state(
    state: Any, mesh: jax.sharding.Mesh
) -> tuple[Any, dict[str, int]]:
    """Change placement of six FP32 arrays, preserving global values and structure."""
    if state.master_params is None:
        raise ValueError("FP32 master weights are required")
    if any(not x.sharding.is_fully_replicated for x in jax.tree.leaves(state.params)):
        raise ValueError("Every compute-model array must remain fully replicated")
    placement = NamedSharding(mesh, P(None, "replica_dcn", None, None))
    counts = {}
    changes = {}
    for name in ("master_params", "opt_state"):
        count = 0

        def partition(path: Any, value: Any) -> Any:
            nonlocal count
            if not _is_expert_array(path, value):
                return value
            if value.dtype != jnp.float32 or value.shape[1] % mesh.size:
                raise ValueError(
                    "Independent expert banks must be FP32 and divide the mesh"
                )
            count += 1
            return jax.sharding.reshard(value, placement)

        changes[name] = jax.tree_util.tree_map_with_path(
            partition, getattr(state, name)
        )
        counts[name] = count
    if counts != {"master_params": 3, "opt_state": 3}:
        raise ValueError(
            f"Expected three expert banks and three momentum buffers: {counts}"
        )
    return dataclasses.replace(state, **changes), counts


def match_gradient_placement(
    optimizer: optax.GradientTransformation,
) -> optax.GradientTransformation:
    """Keep the existing optimizer equations, using the master's explicit layout."""

    def update(grads: Any, state: Any, params: Any = None) -> tuple[Any, Any]:
        if params is None:
            raise ValueError("Explicit master parameter placement is required")

        def place(gradient: Any, param: Any) -> Any:
            if gradient is None:
                return None
            target = _target_named_sharding(param)
            if target is None:
                raise ValueError("A gradient has no matching master-array placement")
            return jax.sharding.reshard(gradient, target)

        placed = jax.tree.map(place, grads, params, is_leaf=lambda x: x is None)
        return optimizer.update(placed, state, params)

    return optax.GradientTransformation(optimizer.init, update)


def sharded_expert_ns(mesh: jax.sharding.Mesh) -> Callable[..., jax.Array]:
    """Run the unchanged matrix polynomial on already-partitioned expert banks."""
    spec = P(None, "replica_dcn", None, None)

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
        if x.ndim != 4 or x.shape[1] % mesh.size:
            raise ValueError("Only independent expert matrices may use this transform")
        if target_sharding is None or target_sharding.spec != spec:
            raise ValueError(
                "Expert optimizer output must preserve its declared partition"
            )

        def local(stack: jax.Array) -> jax.Array:
            flat = stack.reshape((-1, *stack.shape[-2:]))
            if use_syrk:
                # Preserve the deployed cast and normalized BF16 matrix boundary.
                result = _newtonschulz_batched_syrk(
                    flat.astype(jnp.bfloat16), steps, eps, coefficient_type
                )
            else:
                result = jax.vmap(
                    lambda matrix: _zeropower_via_newtonschulz_local(
                        matrix, steps, eps, coefficient_type
                    )
                )(flat)
            return result.reshape(stack.shape).astype(stack.dtype)

        with jax.named_scope("sharded_expert_optimizer"):
            return jax.shard_map(
                local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
            )(jax.sharding.reshard(x, spec))

    return transform


def replicate_model_parameters(state: Any) -> Any:
    """Gather updated BF16 expert parameters; forward/backward see full replicas."""
    params = jax.tree.map(
        lambda x: x if x.sharding.is_fully_replicated else jax.sharding.reshard(x, P()),
        state.params,
    )
    return dataclasses.replace(state, params=params)


def replicate_optimizer_state_for_validation(state: Any) -> Any:
    """Reconstruct canonical global FP32 arrays outside measured training time."""
    return dataclasses.replace(
        state,
        master_params=jax.tree.map(
            lambda x: jax.sharding.reshard(x, P()), state.master_params
        ),
        opt_state=jax.tree.map(lambda x: jax.sharding.reshard(x, P()), state.opt_state),
    )


def assert_partitioned_optimizer_state(
    state: Any, mesh: jax.sharding.Mesh
) -> dict[str, int]:
    """Require complete BF16 model replicas and exactly six partitioned FP32 arrays."""
    if any(not x.sharding.is_fully_replicated for x in jax.tree.leaves(state.params)):
        raise ValueError("Compute model replication was lost")
    counts = {}
    for name in ("master_params", "opt_state"):
        count = 0
        for path, value in jax.tree_util.tree_leaves_with_path(getattr(state, name)):
            if not isinstance(value, jax.Array):
                continue
            if _is_expert_array(path, value):
                if value.dtype != jnp.float32 or value.sharding.spec != P(
                    None, "replica_dcn", None, None
                ):
                    raise ValueError(
                        "Expert FP32 precision or declared partition was lost"
                    )
                count += 1
            elif not value.sharding.is_fully_replicated:
                raise ValueError(
                    "An unrelated optimizer array was unexpectedly partitioned"
                )
        counts[name] = count
    if counts != {"master_params": 3, "opt_state": 3}:
        raise ValueError(f"Unexpected partitioned optimizer structure: {counts}")
    return counts


def persistent_state_bytes(state: Any) -> dict[str, int]:
    """Report logical live array bytes separately from XLA's reserved memory pool."""
    arrays = [x for x in jax.tree.leaves(state) if isinstance(x, jax.Array)]
    return {
        "state_array_leaves": len(arrays),
        "global_array_bytes": sum(x.size * x.dtype.itemsize for x in arrays),
        "per_device_bytes": sum(
            math.prod(x.sharding.shard_shape(x.shape)) * x.dtype.itemsize
            for x in arrays
        ),
    }


def model_matches_fp32_master(params: Any, masters: Any) -> jax.Array:
    """Check the compute model is exactly the BF16 cast of its restored/updated masters."""
    pairs = zip(jax.tree.leaves(params), jax.tree.leaves(masters), strict=True)
    return jnp.all(
        jnp.stack(
            [
                jnp.all(p == jax.sharding.reshard(m.astype(p.dtype), p.sharding))
                for p, m in pairs
            ]
        )
    )
