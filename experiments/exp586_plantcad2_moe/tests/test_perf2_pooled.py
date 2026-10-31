"""Compare the guarded padding path with the unchanged public EP backend."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh

from exp586_moe.perf2_pooled import trimmed_mlp


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
@pytest.mark.parametrize(
    "crowded,capacity,fast_rows,all_padding",
    [
        (False, 8, 2, False),
        (True, 8, 2, False),
        (True, 2, 1, False),
        (False, 8, 2, True),
        (False, 8, (1, 2, 4), False),
        (True, 8, (1, 2, 4), False),
        (True, 2, (1, 2, 4), False),
        (False, 8, (1, 2, 4), True),
    ],
)
def test_padding_fast_path_and_fallback_preserve_output_gradient_and_drops(
    crowded, capacity, fast_rows, all_padding
):
    mesh = compact_grug_mesh(expert_axis_size=8, replica_axis_size=1)
    with jax.set_mesh(mesh):
        expert = MoEExpertMlp.init(
            num_experts=16,
            hidden_dim=4,
            intermediate_dim=8,
            initializer_std=0.1,
            key=jax.random.key(13),
            implementation="fixed_pooled_wave_all_to_all",
            capacity_factor=capacity,
            pooled_transport_capacity_factor=8,
            num_expert_waves=2,
        )
        x = jax.reshard(
            jnp.arange(16 * 4, dtype=jnp.float32).reshape(16, 4) / 64, P("expert", None)
        )
        selected = jnp.tile(jnp.asarray([[0, 1]], dtype=jnp.int32), (16, 1))
        if not crowded:
            selected = (jnp.arange(16)[:, None] + jnp.asarray([[0, 1]])) % 16
        selected = jax.reshard(selected, P("expert", None))
        combine = jax.reshard(jnp.full((16, 2), 0.5), P("expert", None))
        valid = jax.reshard(
            jnp.zeros(16, dtype=bool) if all_padding else jnp.arange(16) % 5 != 0,
            P("expert"),
        )

        def loss(arguments, trimmed):
            weights, inputs, coefficients = arguments
            if trimmed:
                output, counts = trimmed_mlp(
                    weights,
                    inputs,
                    selected,
                    coefficients,
                    token_valid=valid,
                    mesh=mesh,
                    report_capacity_overflow=True,
                    fast_rows=fast_rows,
                )
            else:
                output, counts = weights(
                    inputs,
                    selected,
                    coefficients,
                    token_valid=valid,
                    mesh=mesh,
                    report_capacity_overflow=True,
                )
            return jnp.sum(output**2), (output, counts)

        function = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
        expected, expected_grad = function((expert, x, combine), False)
        actual, actual_grad = function((expert, x, combine), True)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(actual[1][0], expected[1][0], atol=1e-6, rtol=1e-5)
    for left, right in zip(
        jax.tree.leaves(actual[1][1]), jax.tree.leaves(expected[1][1]), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(
        jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True
    ):
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-5)
    if capacity == 2:
        assert int(actual[1][1].dropped) > 0
