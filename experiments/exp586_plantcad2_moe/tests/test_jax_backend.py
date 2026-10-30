import jax
import jax.numpy as jnp
from levanter.grug.sharding import compact_grug_mesh

from exp586_moe.inference.jax_backend import combined_shared_expert_projections


def test_combined_shared_expert_projections_matches_separate_experts() -> None:
    key = jax.random.key(7)
    x_key, *weight_keys = jax.random.split(key, 8)
    x = jax.random.normal(x_key, (13, 8), dtype=jnp.bfloat16)
    experts = []
    for expert in range(2):
        gate = jax.random.normal(weight_keys[3 * expert], (8, 6), dtype=jnp.bfloat16)
        up = jax.random.normal(weight_keys[3 * expert + 1], (8, 6), dtype=jnp.bfloat16)
        down = jax.random.normal(
            weight_keys[3 * expert + 2], (6, 8), dtype=jnp.bfloat16
        )
        experts.append((gate, up, down))

    expected = sum(
        (jax.nn.silu(x @ gate) * (x @ up)) @ down for gate, up, down in experts
    )
    gate_up = jnp.concatenate(
        [weight for gate, up, _ in experts for weight in (gate, up)], axis=-1
    )
    down = jnp.concatenate([down for _, _, down in experts], axis=-2)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        actual = combined_shared_expert_projections(x, gate_up, down, len(experts))

    assert jnp.allclose(actual, expected, atol=0.25, rtol=0.02)
