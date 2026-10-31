import dataclasses
import functools

import jax
import jax.numpy as jnp
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from exp586_moe.replay import independent_copy, three_way_replay
from exp586_moe.routing import damp_qb_updates
from exp586_moe.schedule import TokenClock
from exp586_moe.state import save_dna_checkpoint
from experiments.grug.moe_hero_ep.train import GrugTrainState


def test_live_copies_and_restored_state_survive_donated_updates(tmp_path):
    with jax.set_mesh(compact_grug_mesh()):
        state = GrugTrainState(
            step=jnp.asarray(0, jnp.int32),
            params={"weight": jnp.ones((2, 2))},
            master_params=None,
            opt_state=jnp.zeros((2, 2)),
            ema_params=None,
            pending_qb_betas=jnp.zeros((2, 6)),
        )
        copy = independent_copy(state)
        assert (
            copy.params["weight"].unsafe_buffer_pointer()
            != state.params["weight"].unsafe_buffer_pointer()
        )
        clock = TokenClock()
        path = str(tmp_path / "state")
        save_dna_checkpoint(state, clock, path, data_seed=0)

        @functools.partial(jax.jit, donate_argnums=(0,))
        def step(state, batch):
            amount = jnp.mean(batch.tokens.astype(jnp.float32))
            next_state = dataclasses.replace(
                state,
                step=state.step + 1,
                params={"weight": state.params["weight"] + amount},
                opt_state=state.opt_state + 1,
                pending_qb_betas=state.pending_qb_betas + amount / 10,
            )
            return (
                next_state,
                {
                    "train/loss": amount,
                    "train/router/routing_counts_per_layer": jnp.ones(
                        (2, 6), jnp.int32
                    ),
                    "qb_beta_per_layer": next_state.pending_qb_betas,
                },
                None,
            )

        batches = [
            GrugLmExample(jnp.full((2, 4), i), jnp.ones((2, 4)), AttentionMask.causal())
            for i in (1, 2)
        ]
        logs = []
        result = three_way_replay(
            state, clock, path, jax.sharding.get_mesh(), batches, step, logs.append
        )
    assert result["replay/extra_updates"] == 6
    assert result["replay/max_live_states"] == 2
    assert result["replay/restored_comparison_update"] == 2
    assert "replay/update_1/a_b/params/max_abs" in result
    assert "replay/update_2/a_r/params/max_abs" in result
    assert result["replay/live_copy/layout_equal"]
    for key, value in result.items():
        if key.endswith(("/max_abs", "/relative_l2", "/initial_max_abs")):
            assert value == 0, key


def test_qb_damping_retains_old_thresholds_under_donation():
    state = GrugTrainState(
        step=jnp.asarray(0, jnp.int32),
        params={"weight": jnp.ones(2)},
        master_params=None,
        opt_state=jnp.zeros(2),
        ema_params=None,
        pending_qb_betas=jnp.asarray([1.0, 2.0]),
    )

    @functools.partial(jax.jit, donate_argnums=(0,))
    def native(state, target):
        return (
            dataclasses.replace(state, step=state.step + 1, pending_qb_betas=target),
            {},
            None,
        )

    assert damp_qb_updates(native, 1.0) is native
    wrapped = damp_qb_updates(native, 0.1)
    state, _, _ = wrapped(state, jnp.asarray([11.0, 22.0]))
    assert jnp.allclose(state.pending_qb_betas, jnp.asarray([2.0, 4.0]))
    state, _, _ = wrapped(state, jnp.asarray([12.0, 24.0]))
    assert jnp.allclose(state.pending_qb_betas, jnp.asarray([3.0, 6.0]))
    assert int(state.step) == 2
