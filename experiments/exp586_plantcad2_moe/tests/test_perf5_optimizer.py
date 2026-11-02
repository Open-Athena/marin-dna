"""Check optimizer placement against unchanged independent update oracles."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from exp586_moe.data_parallel import replicated_mesh
from exp586_moe.perf4_muon import node_local_expert_ns
from exp586_moe.perf5_optimizer import (
    match_gradient_placement,
    partition_expert_optimizer_state,
    replicate_model_parameters,
    replicate_optimizer_state_for_validation,
    sharded_expert_ns,
)
from experiments.grug.moe_hero_ep.train import GrugTrainState


def _fixture():
    bank = jax.random.normal(jax.random.key(14), (2, 16, 8, 8))
    masters = {
        "expert_mlp": {
            name: bank + i for i, name in enumerate(("w_gate", "w_up", "w_down"))
        },
        "dense": jnp.arange(32, dtype=jnp.float32).reshape(4, 8),
    }
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.sgd(0.01, momentum=0.95, nesterov=True)
    )
    state = GrugTrainState(
        step=jax.sharding.reshard(jnp.asarray(23), P()),
        params=jax.tree.map(lambda x: x.astype(jnp.bfloat16), masters),
        master_params=masters,
        opt_state=optimizer.init(masters),
        ema_params=None,
        pending_qb_betas=jnp.ones((2, 16)),
    )
    return state, optimizer


def test_partition_preserves_global_state_precision_and_complete_compute_model():
    mesh = replicated_mesh()
    with jax.set_mesh(mesh):
        original, _ = _fixture()
        state, report = partition_expert_optimizer_state(original, mesh)
        assert report == {"master_params": 3, "opt_state": 3}
        assert all(
            x.sharding.is_fully_replicated for x in jax.tree.leaves(state.params)
        )
        for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(state), strict=True):
            assert a.shape == b.shape and a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
        reconstructed = replicate_optimizer_state_for_validation(state)
        for a, b in zip(
            jax.tree.leaves(original), jax.tree.leaves(reconstructed), strict=True
        ):
            np.testing.assert_array_equal(a, b)


def test_global_clipping_and_momentum_match_replica_reference_over_multiple_updates():
    mesh = replicated_mesh()
    with jax.set_mesh(mesh):
        reference, optimizer = _fixture()
        partitioned, _ = partition_expert_optimizer_state(reference, mesh)
        candidate = match_gradient_placement(optimizer)

        @functools.partial(jax.jit, static_argnames=("partitioned",))
        def update(state, gradient, *, partitioned):
            active = candidate if partitioned else optimizer
            changes, moments = active.update(
                gradient, state.opt_state, state.master_params
            )
            masters = optax.apply_updates(state.master_params, changes)
            return dataclasses.replace(
                state,
                master_params=masters,
                opt_state=moments,
                params=jax.tree.map(lambda x: x.astype(jnp.bfloat16), masters),
            )

        for i in range(4):
            # Deliberately uneven expert magnitudes catch accidental per-owner clipping.
            gradients = jax.tree.map(
                lambda x, scale=i + 1: x.astype(jnp.float32) * scale, reference.params
            )
            gradients["expert_mlp"]["w_up"] = (
                gradients["expert_mlp"]["w_up"].at[:, :2].multiply(100)
            )
            reference = update(reference, gradients, partitioned=False)
            partitioned = replicate_model_parameters(
                update(partitioned, gradients, partitioned=True)
            )
            assert all(
                x.sharding.is_fully_replicated
                for x in jax.tree.leaves(partitioned.params)
            )
            for a, b in zip(
                jax.tree.leaves(reference), jax.tree.leaves(partitioned), strict=True
            ):
                np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)


def test_partitioned_matrix_polynomial_matches_deployed_independent_matrix_reference():
    mesh = replicated_mesh()
    if mesh.size != 8:
        pytest.skip("Run with eight virtual CPU devices or the actual eight-GPU mesh")
    with jax.set_mesh(mesh):
        values = jax.random.normal(jax.random.key(73), (2, 16, 8, 8))
        target = NamedSharding(mesh, P(None, "replica_dcn", None, None))
        reference = node_local_expert_ns(mesh, slice_before_cast=True)
        candidate = sharded_expert_ns(mesh)
        expected = jax.jit(
            lambda x: reference(
                None, x, 5, 1e-8, "quintic", False, NamedSharding(mesh, P())
            )
        )(values)
        actual = jax.jit(
            lambda x: candidate(None, x, 5, 1e-8, "quintic", False, target)
        )(values)
        np.testing.assert_array_equal(actual, expected)


def test_real_muon_hyperball_and_all_optimizer_counters_match_global_reference():
    from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig

    mesh = replicated_mesh()
    if mesh.size != 8:
        pytest.skip("Requires the eight-device partitioned oracle")
    with jax.set_mesh(mesh):
        source, _ = _fixture()
        config = GrugMoeMuonHConfig(learning_rate=0.001, adam_lr=0.0002, use_syrk=False)
        reference_opt = config.build(
            100, ns4d_impl=node_local_expert_ns(mesh, slice_before_cast=True)
        )
        candidate_opt = match_gradient_placement(
            config.build(100, ns4d_impl=sharded_expert_ns(mesh))
        )
        reference = dataclasses.replace(
            source, opt_state=reference_opt.init(source.master_params)
        )
        candidate, _ = partition_expert_optimizer_state(reference, mesh)

        def factory(optimizer):
            @jax.jit
            def step(state, gradients):
                updates, moments = optimizer.update(
                    gradients, state.opt_state, state.master_params
                )
                masters = optax.apply_updates(state.master_params, updates)
                return dataclasses.replace(
                    state,
                    master_params=masters,
                    opt_state=moments,
                    params=jax.tree.map(lambda x: x.astype(jnp.bfloat16), masters),
                )

            return step

        step_reference, step_candidate = factory(reference_opt), factory(candidate_opt)
        initial_norms = {
            name: np.linalg.norm(np.asarray(value).reshape(2, -1), axis=1)
            for name, value in source.master_params["expert_mlp"].items()
        }
        for index in range(4):
            gradients = jax.tree.map(
                lambda x, s=index + 1: x.astype(jnp.float32) * s, source.params
            )
            gradients["expert_mlp"]["w_gate"] = (
                gradients["expert_mlp"]["w_gate"].at[:, :2].multiply(100)
            )
            reference = step_reference(reference, gradients)
            candidate = replicate_model_parameters(step_candidate(candidate, gradients))
            for group in ("master_params", "opt_state"):
                for a, b in zip(
                    jax.tree.leaves(getattr(reference, group)),
                    jax.tree.leaves(getattr(candidate, group)),
                    strict=True,
                ):
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
            for name, value in candidate.master_params["expert_mlp"].items():
                norms = np.linalg.norm(np.asarray(value).reshape(2, -1), axis=1)
                np.testing.assert_allclose(norms, initial_norms[name], rtol=2e-6)


def test_compute_weights_must_equal_cast_of_master_weights():
    from exp586_moe.perf5_optimizer import model_matches_fp32_master

    with jax.set_mesh(replicated_mesh()):
        state, _ = _fixture()
        assert bool(model_matches_fp32_master(state.params, state.master_params))
        changed = {**state.params, "dense": state.params["dense"] + 1}
        assert not bool(model_matches_fp32_master(changed, state.master_params))


def test_compute_weight_check_aligns_partitioned_master_sharding():
    from exp586_moe.perf5_optimizer import model_matches_fp32_master

    mesh = replicated_mesh()
    if mesh.size != 8:
        pytest.skip("Requires the eight-device partitioned placement")
    with jax.set_mesh(mesh):
        state, _ = _fixture()
        partitioned, _ = partition_expert_optimizer_state(state, mesh)
        assert bool(
            model_matches_fp32_master(partitioned.params, partitioned.master_params)
        )


def test_compiled_model_gather_matches_external_gather_for_real_optimizer():
    from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig

    mesh = replicated_mesh()
    if mesh.size != 8:
        pytest.skip("Requires the eight-device partitioned oracle")
    with jax.set_mesh(mesh):
        source, _ = _fixture()
        optimizer = match_gradient_placement(
            GrugMoeMuonHConfig(learning_rate=0.001, use_syrk=False).build(
                100, ns4d_impl=sharded_expert_ns(mesh)
            )
        )
        initial = dataclasses.replace(
            source, opt_state=optimizer.init(source.master_params)
        )
        state, _ = partition_expert_optimizer_state(initial, mesh)
        gradients = jax.tree.map(lambda x: x.astype(jnp.float32), source.params)

        @functools.partial(jax.jit, static_argnames=("compiled",))
        def step(state, gradients, *, compiled):
            updates, moments = optimizer.update(
                gradients, state.opt_state, state.master_params
            )
            masters = optax.apply_updates(state.master_params, updates)
            params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), masters)
            if compiled:
                params = jax.tree.map(lambda x: jax.sharding.reshard(x, P()), params)
            return dataclasses.replace(
                state, master_params=masters, params=params, opt_state=moments
            )

        external = replicate_model_parameters(step(state, gradients, compiled=False))
        internal = step(state, gradients, compiled=True)
        assert all(
            x.sharding.is_fully_replicated for x in jax.tree.leaves(internal.params)
        )
        for a, b in zip(
            jax.tree.leaves(external), jax.tree.leaves(internal), strict=True
        ):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
