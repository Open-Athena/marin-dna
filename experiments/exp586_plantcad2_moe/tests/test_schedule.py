import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from exp586_moe.config import (
    PEAK_CHECKPOINT_UPDATE,
    TOKENS_PER_UPDATE,
    TOTAL_UPDATES,
    TRAINING_TOKENS,
    WARMUP_UPDATES,
)
from exp586_moe.schedule import (
    HeroTokenSchedule,
    TokenClock,
    optimizer_config,
    set_token_learning_rates,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic


def test_published_heuristic_reproduces_d768_source_run():
    config = MoeHeuristic().build_optimizer_config(
        num_train_steps=11_420,
        batch_size=1024,
        hidden_dim=768,
        seq_len=4096,
    )
    assert config.adam_lr == pytest.approx(0.003650538063970605, rel=1e-14)
    assert config.learning_rate == pytest.approx(0.01581899827720595, rel=1e-14)
    assert config.beta2 == pytest.approx(0.9684910757595268, rel=1e-14)
    assert config.epsilon == pytest.approx(1.0340199349722422e-15, rel=1e-14)


def test_hero_schedule_uses_ten_epoch_budget_and_one_percent_warmup():
    schedule = HeroTokenSchedule()
    assert TRAINING_TOKENS == 216_158_699_520
    assert TOTAL_UPDATES == 412_290
    assert WARMUP_UPDATES == 4_122
    assert PEAK_CHECKPOINT_UPDATE == 4_123
    assert schedule.factor(0) == 0
    assert schedule.factor((WARMUP_UPDATES - 1) * TOKENS_PER_UPDATE) < 1
    assert schedule.factor(WARMUP_UPDATES * TOKENS_PER_UPDATE) == 1
    assert schedule.factor(PEAK_CHECKPOINT_UPDATE * TOKENS_PER_UPDATE) < 1
    assert schedule.factor((TOTAL_UPDATES - 1) * TOKENS_PER_UPDATE) > 0.05
    assert schedule.factor(TRAINING_TOKENS) == pytest.approx(0.05)
    assert schedule.factor(2 * TRAINING_TOKENS) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="whole update"):
        schedule.factor(1)
    with pytest.raises(ValueError, match="nonnegative"):
        schedule.factor(-TOKENS_PER_UPDATE)


def test_dna_optimizer_uses_actual_training_budget_and_d768_width():
    config = optimizer_config(reference_tokens_per_update=TOKENS_PER_UPDATE)
    assert config.learning_rate == pytest.approx(0.0033199380421085104, rel=1e-14)
    assert config.adam_lr == pytest.approx(0.000766139548178887, rel=1e-14)
    assert config.beta2 == pytest.approx(0.996005996001, rel=1e-14)
    assert config.epsilon == pytest.approx(6.212941441462329e-15, rel=1e-14)
    assert optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE, multiplier=10
    ).learning_rate == pytest.approx(0.033199380421085104, rel=1e-14)


def test_clock_retains_integer_precision():
    clock = TokenClock(216_158_699_517, 216_158_699_500, 17, 9)
    restored = TokenClock(**json.loads(json.dumps(dataclasses.asdict(clock))))
    assert restored.input_tokens == 216_158_699_517
    assert (
        restored.advance(input_tokens=8192, loss_targets=8191, examples=1).input_tokens
        == 216_158_707_709
    )


def test_injected_lr_is_used_inside_norm_preserving_update():
    config = dataclasses.replace(
        optimizer_config(reference_tokens_per_update=65_536), use_syrk=False
    )
    tx = config.build(10)
    params = {
        "output_proj": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "token_embed": jnp.ones((2, 2)),
    }
    grads = jax.tree.map(jnp.ones_like, params)
    schedule = HeroTokenSchedule(
        total_updates=10,
        warmup_updates=2,
        tokens_per_update=100,
        peak_update=3,
    )
    initial = tx.init(params)
    assert not initial.hyperparams_states
    zero = set_token_learning_rates(initial, config, schedule, TokenClock())
    updates, _ = tx.update(grads, zero, params)
    for update in jax.tree.leaves(updates):
        np.testing.assert_allclose(update, 0, atol=5e-7)
    peak = set_token_learning_rates(
        initial,
        config,
        schedule,
        TokenClock(input_tokens=200),
    )
    updates, after = tx.update(grads, peak, params)
    moved = optax.apply_updates(params, updates)
    assert not np.allclose(moved["output_proj"], params["output_proj"])
    np.testing.assert_allclose(
        np.linalg.norm(moved["output_proj"]),
        np.linalg.norm(params["output_proj"]),
        rtol=1e-6,
    )
    assert float(after.hyperparams["learning_rate"]) == pytest.approx(
        config.learning_rate
    )
    assert float(after.hyperparams["adam_lr"]) == pytest.approx(config.adam_lr)
