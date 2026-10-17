import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from test_model_contract import tiny_config

from exp586_moe.precision import RouterFloat32Policy
from experiments.grug.moe_hero_ep.model import Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.train import _loss_and_grads


def test_router_policy_preserves_small_bias_differences_and_weight_updates():
    base = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    precise = RouterFloat32Policy(
        base.param_dtype, base.compute_dtype, base.output_dtype
    )
    with jax.set_mesh(compact_grug_mesh()):
        master = Transformer.init(tiny_config(), key=jax.random.key(0))
        original_router = master.stacked_blocks.stacked.mlp.router
        # Updates smaller than a BF16 ULP must survive master -> parameter -> compute.
        master = eqx.tree_at(
            lambda m: m.stacked_blocks.stacked.mlp.router,
            master,
            jnp.ones_like(original_router) + 0.0001,
        )
        params = precise.cast_to_param(master)
        preserved = {
            id(params.stacked_blocks.stacked.mlp.router),
            id(params.stacked_blocks.stacked.mlp.router_bias),
        }
        for value in jax.tree.leaves(params):
            assert value.dtype == (
                jnp.float32 if id(value) in preserved else jnp.bfloat16
            )
        betas = jnp.tile(jnp.asarray([1.0001, 1.0002, 1.0003, -1, -1, -1]), (2, 1))
        applied = apply_qb_betas(params, betas)
        compute = precise.cast_to_compute(applied)
        rounded = base.cast_to_compute(applied)
    router = compute.stacked_blocks.stacked.mlp
    np.testing.assert_array_equal(
        router.router, master.stacked_blocks.stacked.mlp.router
    )
    np.testing.assert_array_equal(
        router.router_bias, applied.stacked_blocks.stacked.mlp.router_bias
    )
    assert router.router.dtype == router.router_bias.dtype == jnp.float32
    assert compute.token_embed.dtype == jnp.bfloat16
    assert np.unique(np.asarray(router.router_bias[0, :3])).size == 3
    assert (
        np.unique(
            np.asarray(rounded.stacked_blocks.stacked.mlp.router_bias[0, :3])
        ).size
        == 1
    )
    logits = jnp.einsum("td,de->te", jnp.ones((2, 32), jnp.bfloat16), router.router[0])
    assert logits.dtype == jnp.float32


def test_precise_router_runs_native_loss_and_gradients():
    policy = RouterFloat32Policy(jnp.bfloat16, jnp.bfloat16, jnp.bfloat16)
    with jax.set_mesh(compact_grug_mesh()):
        params = policy.cast_to_param(
            Transformer.init(tiny_config(), key=jax.random.key(1))
        )
        tokens = jnp.asarray([[3, 4, 5, 6] * 4, [6, 5, 4, 3] * 4])
        batch = GrugLmExample(
            tokens, jnp.ones_like(tokens, dtype=jnp.float32), AttentionMask.causal()
        )
        (loss, metrics), grads = eqx.filter_jit(
            lambda model: _loss_and_grads(model, batch, policy, 1e-4)
        )(params)
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(grads))
    assert grads.stacked_blocks.stacked.mlp.router.dtype == jnp.float32
    assert metrics["qb_beta_per_layer"].dtype == jnp.float32
