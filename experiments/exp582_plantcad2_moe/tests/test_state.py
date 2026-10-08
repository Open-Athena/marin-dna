import dataclasses

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from levanter.grug.sharding import compact_grug_mesh
from test_model_contract import tiny_config

from exp582_moe.schedule import (
    TokenClock,
    TokenSchedule,
    optimizer_config,
    set_token_learning_rates,
)
from exp582_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_all_finite,
    state_error_by_group,
    state_max_abs_error,
)
from experiments.grug.moe_hero_ep.model import Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.train import GrugTrainState


def test_transfer_retains_pending_bias_and_checkpoint_retains_exact_clock(tmp_path):
    with jax.set_mesh(compact_grug_mesh()):
        weights = Transformer.init(tiny_config(), key=jax.random.key(0))
        weights = apply_qb_betas(
            weights, jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
        )
        tx = dataclasses.replace(
            optimizer_config(reference_tokens_per_update=65536), use_syrk=False
        ).build(10)
        state = fresh_state(weights, tx, jmp.get_policy("float32"), offload=False)
        assert bool(state_all_finite(state))
        invalid = dataclasses.replace(
            state, pending_qb_betas=state.pending_qb_betas.at[0, 0].set(jnp.nan)
        )
        assert not bool(state_all_finite(invalid))
        effective = apply_qb_betas(state.params, state.pending_qb_betas)
        np.testing.assert_array_equal(
            effective.stacked_blocks.stacked.mlp.router_bias,
            weights.stacked_blocks.stacked.mlp.router_bias,
        )
        clock = TokenClock(216_158_699_517, 216_158_699_500, 17, 0)
        path = str(tmp_path / "checkpoint")
        save_dna_checkpoint(state, clock, path, data_seed=0, tokenizer_sha256="a" * 64)
        with pytest.raises(ValueError, match="tokenizer digest mismatch"):
            restore_dna_checkpoint(
                state,
                path,
                jax.sharding.get_mesh(),
                data_seed=0,
                tokenizer_sha256="b" * 64,
            )
        restored, restored_clock = restore_dna_checkpoint(
            state, path, jax.sharding.get_mesh(), data_seed=0, tokenizer_sha256="a" * 64
        )
        assert restored_clock == clock
        assert float(state_max_abs_error(state, restored)) == 0
        assert all(
            float(value) == 0
            for value in state_error_by_group(state, restored).values()
        )
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            np.testing.assert_array_equal(a, b)


def test_native_optimizer_forks_from_checkpoint_without_rewarming(tmp_path):
    """Restore nonzero moments, then exercise both sides of the cooldown branch."""
    with jax.set_mesh(compact_grug_mesh()):
        cfg = dataclasses.replace(
            optimizer_config(reference_tokens_per_update=32), use_syrk=False
        )
        tx = cfg.build(10)
        # Exercise all three actual optimizer groups with a small deterministic objective.
        params = {
            "attn": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "output_proj": jnp.asarray([[1.0, -2.0], [3.0, -4.0]]),
            "token_embed": jnp.ones((2, 2)),
        }
        state = GrugTrainState(
            step=jnp.asarray(0, dtype=jnp.int32),
            params=params,
            master_params=None,
            opt_state=tx.init(params),
            ema_params=None,
            pending_qb_betas=jnp.zeros((2, 6)),
        )
        cooling = TokenSchedule(horizon=640, cooldown_start=64, stage_end=128)
        continuing = dataclasses.replace(cooling, cooldown=False)

        def step(state, clock, schedule):
            state = dataclasses.replace(
                state,
                opt_state=set_token_learning_rates(
                    state.opt_state, cfg, schedule, clock
                ),
            )
            grads = jax.tree.map(lambda p: jnp.cos(p), state.params)
            updates, opt_state = tx.update(grads, state.opt_state, state.params)
            state = dataclasses.replace(
                state,
                step=state.step + 1,
                params=optax.apply_updates(state.params, updates),
                opt_state=opt_state,
            )
            return state, clock.advance(input_tokens=32, loss_targets=30, examples=2)

        clock = TokenClock()
        for _ in range(2):
            state, clock = step(state, clock, continuing)
        assert clock.input_tokens == cooling.cooldown_start
        path = str(tmp_path / "branch")
        save_dna_checkpoint(state, clock, path, data_seed=0)
        # Restore before consuming/donating either branch's arrays.
        fork, fork_clock = restore_dna_checkpoint(
            state, path, jax.sharding.get_mesh(), data_seed=0
        )
        state, clock = step(state, clock, continuing)
        fork, fork_clock = step(fork, fork_clock, cooling)
        assert clock == fork_clock
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(fork), strict=True):
            np.testing.assert_array_equal(a, b)
        state, clock = step(state, clock, continuing)
        fork, fork_clock = step(fork, fork_clock, cooling)
        assert clock == fork_clock
        assert clock.input_tokens == cooling.stage_end
        # Both branches retained their optimizer counters; their actual updates now differ.
        assert int(state.opt_state.count) == int(fork.opt_state.count) == 4
        assert float(state.opt_state.hyperparams["adam_lr"]) == pytest.approx(
            cfg.adam_lr * continuing.factor(96)
        )
        assert float(fork.opt_state.hyperparams["adam_lr"]) == pytest.approx(
            cfg.adam_lr * cooling.factor(96)
        )
        for group in params:
            assert not np.array_equal(state.params[group], fork.params[group])
