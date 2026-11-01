"""Compare replicated local MoE with zero-drop EP on identical assignments."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh

from exp586_moe.data_parallel import (
    assert_physical_replicas,
    assert_replicated_state,
    local_model,
    physical_projection_snapshot,
    replica_fingerprints,
    replicated_mesh,
)
from exp586_moe.state import fresh_state
from experiments.grug.moe_hero_ep.grugmuon_hero import (
    _intra_rack_axes,
    _newtonschulz_4d_distributed,
)
from experiments.grug.moe_hero_ep.model import (
    GrugModelConfig,
    Transformer,
    _qb_beta_hist,
)


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
@pytest.mark.parametrize(
    "crowded,all_padding", [(False, False), (True, False), (False, True)]
)
def test_local_replication_preserves_zero_drop_ep_forward_and_gradients(
    crowded, all_padding
):
    outputs = []
    for local in (False, True):
        mesh = (
            replicated_mesh()
            if local
            else compact_grug_mesh(expert_axis_size=8, replica_axis_size=1)
        )
        with jax.set_mesh(mesh):
            expert = MoEExpertMlp.init(
                num_experts=16,
                hidden_dim=8,
                intermediate_dim=16,
                initializer_std=0.1,
                key=jax.random.key(17),
                implementation="scatter" if local else "fixed_pooled_wave_all_to_all",
                capacity_factor=32,
                pooled_transport_capacity_factor=None if local else 8,
                num_expert_waves=1 if local else 2,
            )
            if local:
                assert all(
                    x.sharding.is_fully_replicated for x in jax.tree.leaves(expert)
                )
            token_axis = "replica_dcn" if local else "expert"
            x = jax.device_put(
                jnp.arange(64 * 8, dtype=jnp.float32).reshape(64, 8) / 512,
                NamedSharding(mesh, P(token_axis, None)),
            )
            selected = (
                jnp.tile(jnp.asarray([[0, 1]], jnp.int32), (64, 1))
                if crowded
                else (jnp.arange(64)[:, None] + jnp.asarray([[0, 1]])) % 16
            )
            selected = jax.device_put(
                selected, NamedSharding(mesh, P(token_axis, None))
            )
            combine = jax.device_put(
                jnp.full((64, 2), 0.5), NamedSharding(mesh, P(token_axis, None))
            )
            valid = jax.device_put(
                jnp.zeros(64, bool) if all_padding else jnp.arange(64) % 5 != 0,
                NamedSharding(mesh, P(token_axis)),
            )

            def loss(arguments, selected=selected, valid=valid, mesh=mesh):
                weights, inputs, coefficients = arguments
                output, counts = weights(
                    inputs,
                    selected,
                    coefficients,
                    token_valid=valid,
                    mesh=mesh,
                    report_capacity_overflow=True,
                )
                return jnp.mean(output**2), (output, counts)

            value, gradient = eqx.filter_jit(
                eqx.filter_value_and_grad(loss, has_aux=True)
            )((expert, x, combine))
            outputs.append((jax.device_get(value), jax.device_get(gradient)))
    for left, right in zip(
        jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]), strict=True
    ):
        np.testing.assert_allclose(left, right, atol=2e-7, rtol=2e-5)
    assert int(outputs[0][0][1][1].dropped) == int(outputs[1][0][1][1].dropped) == 0


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
def test_full_model_and_optimizer_are_replicated():
    mesh = replicated_mesh()
    assert mesh.shape == {
        "replica_dcn": 8,
        "data": 1,
        "context": 1,
        "expert": 1,
        "model": 1,
    }
    source = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=16,
        num_experts=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
        sliding_window=8,
        moe_implementation="fixed_pooled_wave_all_to_all",
        pooled_transport_capacity_factor=8,
        capacity_factor=32,
        num_expert_waves=3,
    )
    cfg = local_model(source)
    assert cfg.vocab_size == source.vocab_size and cfg.num_experts == source.num_experts
    with jax.set_mesh(mesh):
        weights = eqx.filter_jit(Transformer.init)(cfg, key=jax.random.key(0))
        state = eqx.filter_jit(
            lambda w: fresh_state(
                w,
                optax.adamw(1e-3),
                jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16"),
                offload=False,
                retain_master=True,
            )
        )(weights)
        sizes = assert_replicated_state(state)
    assert sizes["master_params"] == 2 * sizes["params"]
    assert sizes["opt_state"] > sizes["master_params"]


class _Projection(eqx.Module):
    output_proj: jax.Array


class _ReplicaState(eqx.Module):
    params: _Projection
    master_params: _Projection


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
def test_physical_replica_checks_observe_each_gpu_copy():
    mesh = replicated_mesh()
    sharding = NamedSharding(mesh, P())
    values = np.random.default_rng(86).normal(size=(8, 16)).astype(np.float32)
    good = jax.device_put(values, sharding)
    state = _ReplicaState(_Projection(good.astype(jnp.bfloat16)), _Projection(good))
    with jax.set_mesh(mesh):
        fingerprints = replica_fingerprints(state, mesh)
        assert fingerprints.shape == (8, 6)
        report = assert_physical_replicas(state, mesh)
        assert report["arrays_checked"] == 2
        assert report["master_projection_norm_min"] == pytest.approx(
            report["master_projection_norm_max"]
        )
        snapshot = physical_projection_snapshot(good, mesh)
        assert len(set(snapshot["sha256_by_rank"])) == 1

        buffers = [
            jax.device_put(values + (index == 5), device)
            for index, device in enumerate(mesh.devices.reshape(-1))
        ]
        inconsistent = jax.make_array_from_single_device_arrays(
            values.shape, sharding, buffers
        )
        bad = _ReplicaState(
            _Projection(inconsistent.astype(jnp.bfloat16)),
            _Projection(inconsistent),
        )
        with pytest.raises(ValueError, match="differ across physical replicas"):
            assert_physical_replicas(bad, mesh)


def test_only_local_backends_are_allowed():
    with pytest.raises(ValueError, match="local backends"):
        local_model(GrugModelConfig(vocab_size=32), implementation="ring")


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
def test_histogram_router_is_independent_of_replica_partition():
    results = []
    for device_count in (1, 8):
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:device_count]).reshape(device_count, 1, 1, 1, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            margins = jax.device_put(
                np.random.default_rng(12).normal(size=(512, 16)).astype(np.float32),
                NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None)),
            )
            valid = jax.device_put(
                np.arange(512) % 7 != 0,
                NamedSharding(mesh, P(("replica_dcn", "data", "expert"))),
            )
            fn = jax.jit(
                lambda x, mask, mesh=mesh: _qb_beta_hist(
                    x, mask, mesh, num_experts_per_token=2, num_experts=16, n_bins=1000
                )
            )
            results.append(jax.device_get(fn(margins, valid)))
    for a, b in zip(results[0], results[1], strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
@pytest.mark.parametrize("random_inputs", [False, True])
def test_local_optimizer_math_preserves_ep_and_uses_no_model_collectives(random_inputs):
    outputs = []
    for local in (False, True):
        mesh = (
            replicated_mesh()
            if local
            else compact_grug_mesh(expert_axis_size=8, replica_axis_size=1)
        )
        with jax.set_mesh(mesh):
            if local:
                assert _intra_rack_axes(mesh) == []
            spec = P(None, "expert", "data", "model")
            x = jax.device_put(
                (
                    jax.random.normal(jax.random.key(8), (2, 16, 8, 8))
                    if random_inputs
                    else jnp.broadcast_to(jnp.eye(8, dtype=jnp.float32), (2, 16, 8, 8))
                    * jnp.arange(1, 33, dtype=jnp.float32).reshape(2, 16, 1, 1)
                ),
                NamedSharding(mesh, spec),
            )
            fn = jax.jit(
                lambda value, mesh=mesh, spec=spec: _newtonschulz_4d_distributed(
                    (),
                    value,
                    5,
                    1e-8,
                    "quintic",
                    False,
                    target_sharding=NamedSharding(mesh, spec),
                )
            )
            outputs.append(np.asarray(fn(x)))
            if local:
                hlo = fn.lower(x).compile().as_text()
                assert (
                    "all-gather" not in hlo
                    and "all-to-all" not in hlo
                    and "reduce-scatter" not in hlo
                )
    if random_inputs:
        # BF16 Newton-Schulz amplifies contraction-rounding differences; test
        # direction and norm, then calibrate actual update/loss drift on H100s.
        a, b = (x.reshape(-1).astype(np.float64) for x in outputs)
        assert np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.999
        assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.02
    else:
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-6, rtol=1e-6)
