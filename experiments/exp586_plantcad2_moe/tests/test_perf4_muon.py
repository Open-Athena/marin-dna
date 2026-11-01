"""Check independent-matrix partitioning and complete replica reconstruction."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from exp586_moe import perf4_muon
from exp586_moe.config import TOKENS_PER_UPDATE
from exp586_moe.data_parallel import replicated_mesh
from exp586_moe.perf4_muon import node_local_expert_ns
from exp586_moe.schedule import optimizer_config
from experiments.grug.moe_hero_ep import grugmuon_hero
from experiments.grug.moe_hero_ep.grugmuon_hero import _newtonschulz_4d_distributed


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
@pytest.mark.parametrize("group_size", [4, 8])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("annotated", [False, True])
def test_local_matrix_work_matches_reference_on_every_replica(
    group_size, dtype, annotated
):
    mesh = replicated_mesh()
    values = np.random.default_rng(84).normal(size=(2, 8, 8, 12)).astype(np.float32)
    values[0, 0] = 0
    transform = node_local_expert_ns(mesh, group_size=group_size)
    with jax.set_mesh(mesh):
        spec = P(None, ("expert", "context"), "data", "model") if annotated else P()
        x = jax.device_put(values.astype(dtype), NamedSharding(mesh, spec))
        reference = jax.jit(
            lambda a: _newtonschulz_4d_distributed((), a, 5, 1e-8, "quintic", False)
        )(x)
        result = jax.jit(
            lambda a: transform((), a, 5, 1e-8, "quintic", False, x.sharding)
        )(x)
        assert result.sharding.is_fully_replicated
        for shard in result.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), np.asarray(reference))


def test_optimizer_groups_cannot_cross_node_boundaries():
    fake = SimpleNamespace(
        shape={"replica_dcn": 8, "data": 1},
        devices=np.asarray([SimpleNamespace(process_index=i * 8) for i in range(8)]),
    )
    with pytest.raises(ValueError, match="cross physical"):
        node_local_expert_ns(fake)


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
def test_quack_wrapper_preserves_the_reference_input_and_return_dtype(monkeypatch):
    def cpu_batched(stack, steps, eps, coefficient_type):
        assert stack.dtype == jnp.bfloat16
        return jax.vmap(
            lambda a: grugmuon_hero._zeropower_via_newtonschulz_local(
                a, steps, eps, coefficient_type
            )
        )(stack)

    monkeypatch.setattr(grugmuon_hero, "_newtonschulz_batched_syrk", cpu_batched)
    monkeypatch.setattr(perf4_muon, "_newtonschulz_batched_syrk", cpu_batched)
    mesh = replicated_mesh()
    transform = node_local_expert_ns(mesh)
    with jax.set_mesh(mesh):
        x = jax.device_put(
            np.random.default_rng(85).normal(size=(2, 8, 8, 12)).astype(np.float32),
            NamedSharding(mesh, P()),
        )
        reference = jax.jit(
            lambda a: _newtonschulz_4d_distributed((), a, 5, 1e-8, "quintic", True)
        )(x)
        result = jax.jit(lambda a: transform((), a, 5, 1e-8, "quintic", True))(x)
        assert result.dtype == reference.dtype == jnp.float32
        for shard in result.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), np.asarray(reference))


def test_runtime_implementation_preserves_optimizer_structure_and_equations():
    calls = []

    def same_formula(*args, **kwargs):
        calls.append(True)
        return _newtonschulz_4d_distributed(*args, **kwargs)

    config = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE,
        training_tokens=1_000_000_192_512,
    )
    original = config.build(1_907_349)
    candidate = config.build(1_907_349, ns4d_impl=same_formula)
    params = {
        "w13": jnp.asarray(
            np.random.default_rng(83).normal(size=(2, 4, 8, 12)), dtype=jnp.float32
        )
    }
    left, right = original.init(params), candidate.init(params)
    a, b = params, params
    for _ in range(3):
        grads = jax.tree.map(jnp.sin, a)
        ua, left = original.update(grads, left, a)
        ub, right = candidate.update(grads, right, b)
        a, b = optax.apply_updates(a, ua), optax.apply_updates(b, ub)
        assert jax.tree.structure(left) == jax.tree.structure(right)
        for first, second in zip(
            jax.tree.leaves((a, left)), jax.tree.leaves((b, right)), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert calls


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
def test_full_stack_normalization_matches_the_materialized_reference(monkeypatch):
    # Isolate the BF16 boundary consumed by the first real QuACK call.
    monkeypatch.setattr(
        perf4_muon, "_iterate_normalized_syrk", lambda x, steps, coefficients: x
    )
    mesh = replicated_mesh()
    transform = node_local_expert_ns(mesh, normalize_before_split=True)
    values = np.random.default_rng(87).normal(size=(2, 8, 8, 12)).astype(np.float32)
    values[0, 0] = 0

    def reference(x):
        flat = x.astype(jnp.bfloat16).reshape(-1, 8, 12)
        normalized = flat / (jnp.linalg.norm(flat, axis=(-2, -1), keepdims=True) + 1e-8)
        return jax.lax.optimization_barrier(normalized).reshape(x.shape).astype(x.dtype)

    with jax.set_mesh(mesh):
        x = jax.device_put(values, NamedSharding(mesh, P()))
        expected = jax.jit(reference)(x)
        actual = jax.jit(lambda a: transform((), a, 0, 1e-8, "quintic", True))(x)
        for shard in actual.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), np.asarray(expected))


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
@pytest.mark.parametrize("slice_before_cast", [False, True])
@pytest.mark.parametrize("cast_before_gather", [False, True])
def test_boundary_reordering_preserves_materialized_bf16_bytes(
    monkeypatch, slice_before_cast, cast_before_gather
):
    # Model the physical BF16 operand/result boundary of a QuACK custom call.
    def materialized(stack, steps, eps, coefficient_type):
        assert stack.dtype == jnp.bfloat16
        return jax.lax.optimization_barrier(stack)

    monkeypatch.setattr(perf4_muon, "_newtonschulz_batched_syrk", materialized)
    mesh = replicated_mesh()
    transform = node_local_expert_ns(
        mesh,
        slice_before_cast=slice_before_cast,
        cast_before_gather=cast_before_gather,
    )
    values = np.random.default_rng(88).normal(size=(2, 8, 8, 12)).astype(np.float32)
    expected = np.asarray(values.astype(jnp.bfloat16).astype(jnp.float32))
    with jax.set_mesh(mesh):
        x = jax.device_put(values, NamedSharding(mesh, P()))
        actual = jax.jit(lambda a: transform((), a, 5, 1e-8, "quintic", True))(x)
        for shard in actual.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), expected)
