"""Exercise the real model at reduced size; full GPU kernels have a separate gate."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from exp582_moe.config import d1536_config
from experiments.grug.moe_hero_ep.model import Transformer


def tiny_config():
    # High capacity removes token-dropping as a confound in the padding oracle.
    return dataclasses.replace(
        d1536_config("random"),
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        latent_dim=16,
        num_experts=6,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=32,
        sliding_window=8,
        global_every=2,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        capacity_factor=6.0,
    )


def test_padding_keeps_valid_hidden_states_and_qb_statistics():
    cfg = tiny_config()
    tokens = np.asarray([[3, 4, 5, 6] * 4, [6, 5, 4, 3] * 4], dtype=np.int32)
    padded = np.pad(tokens, ((0, 0), (0, 16)))
    segments = np.broadcast_to(
        np.where(np.arange(32)[None, :] < 16, 0, -1), (2, 32)
    ).astype(np.int32)
    with jax.set_mesh(compact_grug_mesh()):
        model = Transformer.init(cfg, key=jax.random.key(1))
        forward = eqx.filter_jit(lambda m, ids, mask: m(ids, mask))
        compact_hidden, compact_metrics = forward(
            model, jnp.asarray(tokens), AttentionMask.causal()
        )
        padded_hidden, padded_metrics = forward(
            model,
            jnp.asarray(padded),
            AttentionMask.causal().with_segment_ids(jnp.asarray(segments)),
        )
    np.testing.assert_allclose(
        padded_hidden[:, :16], compact_hidden, atol=1e-5, rtol=1e-5
    )
    for key in (
        "qb_beta_per_layer",
        "routing_counts_per_layer",
        "router_z_loss_per_layer",
    ):
        np.testing.assert_allclose(
            padded_metrics[key], compact_metrics[key], atol=1e-5, rtol=1e-5
        )
    np.testing.assert_array_equal(padded_metrics["capacity_overflow_per_layer"], 0)
    np.testing.assert_array_equal(
        padded_metrics["skipped_assignments_per_layer"], 32 * cfg.num_experts_per_token
    )


def test_model_parameter_shapes_do_not_depend_on_context_length():
    with jax.set_mesh(compact_grug_mesh()):
        cfg = d1536_config("pretrained")
        full = eqx.filter_eval_shape(Transformer.init, cfg, key=jax.random.key(0))
        short = eqx.filter_eval_shape(
            Transformer.init,
            dataclasses.replace(cfg, max_seq_len=4096),
            key=jax.random.key(0),
        )
        scratch = eqx.filter_eval_shape(
            Transformer.init, d1536_config("random"), key=jax.random.key(0)
        )
    shapes = lambda model: [leaf.shape for leaf in jax.tree.leaves(model)]
    assert shapes(full) == shapes(short)
    full_count = sum(x.size for x in jax.tree.leaves(full))
    scratch_count = sum(x.size for x in jax.tree.leaves(scratch))
    assert full_count - scratch_count == 2 * 1536 * (128256 - 8)
    assert cfg.rope.theta == 10000
    assert cfg.rope_fused and cfg.qk_mult == 1.3
