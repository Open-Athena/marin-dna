"""Run with XLA_FLAGS=--xla_force_host_platform_device_count=8 for the CPU oracle."""

import dataclasses

import equinox as eqx
import jax
import jmp
import numpy as np
import pytest
from levanter.callbacks.watch import WatchConfig
from levanter.grug.sharding import compact_grug_mesh
from test_model_contract import tiny_config

from exp586_moe.data import WindowBatch
from exp586_moe.gpu_smoke import device_batch
from exp586_moe.placement_smoke import local_state_digest
from exp586_moe.schedule import TokenClock, optimizer_config
from exp586_moe.state import fresh_state, restore_dna_checkpoint, save_dna_checkpoint
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import _make_train_step


@pytest.mark.skipif(jax.device_count() != 8, reason="Requires eight CPU devices")
def test_context_split_preserves_resumed_update_and_global_routing(tmp_path):
    cfg = dataclasses.replace(tiny_config(), num_experts=8, capacity_factor=8)
    policy = jmp.get_policy("float32")
    optimizer = dataclasses.replace(
        optimizer_config(reference_tokens_per_update=256), use_syrk=False
    ).build(10)
    ids = np.random.default_rng(4).integers(3, 7, size=(8, 32), dtype=np.int32)
    weights = np.ones((8, 32), dtype=np.float32)
    weights[:, -1] = 0
    batch = WindowBatch(
        ids, np.zeros_like(ids), weights, np.full(8, 32, dtype=np.int32)
    )
    clock = TokenClock(input_tokens=256, loss_targets=248, examples=8, updates=1)
    results = []
    for context_size in (1, 2):
        mesh = compact_grug_mesh(
            expert_axis_size=2,
            replica_axis_size=4 // context_size,
            context_axis_size=context_size,
        )
        with jax.set_mesh(mesh):
            state = eqx.filter_jit(
                lambda: fresh_state(
                    Transformer.init(cfg, key=jax.random.key(0)),
                    optimizer,
                    policy,
                    offload=False,
                )
            )()
            step = _make_train_step(
                optimizer,
                policy,
                z_loss_weight=1e-4,
                ema_beta=None,
                offload_opt_state=False,
                watch_config=WatchConfig(
                    watch_targets=["grads", "updates"],
                    include_per_parameter_norms=False,
                    split_scan_layers=False,
                    include_histograms=False,
                ),
            )
            inputs = device_batch(batch, mesh)
            if context_size == 1:
                state, _, _ = step(state, inputs)
                save_dna_checkpoint(state, clock, str(tmp_path / "state"), data_seed=0)
            else:
                state, restored_clock = restore_dna_checkpoint(
                    state, str(tmp_path / "state"), mesh, data_seed=0
                )
                assert restored_clock == clock
            state, metrics, watch = step(state, inputs)
            jax.block_until_ready((state, metrics, watch))
            if context_size == 2:
                saved_clock = clock.advance(
                    input_tokens=256, loss_targets=248, examples=8
                )
                checkpoint = str(tmp_path / "context-state")
                save_dna_checkpoint(state, saved_clock, checkpoint, data_seed=0)
                original_digest = local_state_digest(state)
                template = jax.tree.map(
                    lambda a: jax.ShapeDtypeStruct(
                        a.shape, a.dtype, sharding=a.sharding
                    ),
                    state,
                )
                restored, restored_clock = restore_dna_checkpoint(
                    template, checkpoint, mesh, data_seed=0
                )
                assert restored_clock == saved_clock
                assert local_state_digest(restored) == original_digest
            results.append(jax.tree.map(np.asarray, (state, metrics, watch)))
    for first, second in zip(
        jax.tree.leaves(results[0][0]), jax.tree.leaves(results[1][0]), strict=True
    ):
        np.testing.assert_allclose(first, second, atol=1e-5, rtol=1e-4)
    for key in (
        "train/loss",
        "train/cross_entropy_loss",
        "train/router/routing_counts_per_layer",
        "qb_beta_per_layer",
        "moe/valid_assignments",
        "moe/dropped_assignments",
    ):
        np.testing.assert_allclose(
            results[0][1][key], results[1][1][key], atol=1e-5, rtol=1e-4
        )
    np.testing.assert_array_equal(results[1][1]["moe/dropped_assignments"], 0)
    for key in ("grad/norm/total", "updates/norm/total"):
        np.testing.assert_allclose(
            results[0][2][key], results[1][2][key], atol=1e-5, rtol=1e-4
        )
