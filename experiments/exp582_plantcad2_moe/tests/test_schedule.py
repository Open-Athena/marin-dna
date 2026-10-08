import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from exp582_moe.config import HORIZON_TOKENS, SHORT_STAGE_TOKENS, TOKENS_PER_UPDATE
from exp582_moe.schedule import (
    ShortTokenSchedule,
    TokenClock,
    TokenSchedule,
    optimizer_config,
    set_token_learning_rates,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic


def test_published_heuristic_reproduces_source_run():
    config = MoeHeuristic().build_optimizer_config(
        num_train_steps=15128, batch_size=6144, hidden_dim=1536, seq_len=4096
    )
    assert config.adam_lr == pytest.approx(0.0034360151656866902, rel=1e-14)
    assert config.learning_rate == pytest.approx(0.01488939905130899, rel=1e-14)
    assert config.beta2 == 0.95
    assert config.epsilon == pytest.approx(1.1901086660166794e-15, rel=1e-14)


def test_short_schedule_preserves_history_and_places_peak_before_permanent_save():
    original, short = TokenSchedule(), ShortTokenSchedule()
    assert SHORT_STAGE_TOKENS == 11_240_734_720
    for update in range(10719):
        tokens = update * TOKENS_PER_UPDATE
        assert short.factor(tokens) == original.factor(tokens)
    assert short.factor(10719 * TOKENS_PER_UPDATE) == 1
    assert short.factor(10720 * TOKENS_PER_UPDATE) < 1
    assert short.factor(16079 * TOKENS_PER_UPDATE) == pytest.approx(0.525)
    assert short.factor(21439 * TOKENS_PER_UPDATE) == pytest.approx(0.05)
    assert short.factor(SHORT_STAGE_TOKENS) == pytest.approx(0.05)
    assert short.factor(2 * SHORT_STAGE_TOKENS) == pytest.approx(0.05)
    # Explicit late bridge: restore 10771, perform one peak update, save 10772.
    late = ShortTokenSchedule(peak_update=10772)
    assert late.factor(10771 * TOKENS_PER_UPDATE) == 1
    assert late.factor(10773 * TOKENS_PER_UPDATE) < 1
    assert late.stage_end == short.stage_end
    assert late.factor(late.stage_end) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ShortTokenSchedule(peak_update=21440)
    with pytest.raises(ValueError):
        short.factor(-1)
    opt = optimizer_config(reference_tokens_per_update=TOKENS_PER_UPDATE)
    assert opt.learning_rate == pytest.approx(0.00187806248, rel=1e-8)
    assert opt.adam_lr == pytest.approx(0.000433399034, rel=1e-8)
    assert opt.epsilon == pytest.approx(1.00181512e-14, rel=1e-8)


def test_schedule_branches_without_new_warmup_and_clock_retains_integer_precision():
    schedule = TokenSchedule()
    continuing = dataclasses.replace(schedule, cooldown=False)
    assert schedule.factor(0) == 0
    assert schedule.factor(round(HORIZON_TOKENS / 100)) == pytest.approx(1)
    assert schedule.factor(schedule.cooldown_start) == continuing.factor(
        schedule.cooldown_start
    )
    assert schedule.factor(schedule.stage_end) == pytest.approx(0.05)
    assert continuing.factor(schedule.stage_end) > 0.6
    assert continuing.factor(schedule.horizon) == pytest.approx(0.05)
    clock = TokenClock(216_158_699_517, 216_158_699_500, 17, 9)
    restored = TokenClock(**json.loads(json.dumps(dataclasses.asdict(clock))))
    assert restored.input_tokens == 216_158_699_517
    assert (
        restored.advance(input_tokens=8192, loss_targets=8191, examples=1).input_tokens
        == 216_158_707_709
    )


def test_injected_lr_is_used_inside_norm_preserving_update():
    config = dataclasses.replace(
        optimizer_config(reference_tokens_per_update=65536), use_syrk=False
    )
    tx = config.build(100)
    params = {
        "output_proj": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "token_embed": jnp.ones((2, 2)),
    }
    grads = jax.tree.map(jnp.ones_like, params)
    schedule = TokenSchedule(horizon=10000, stage_end=4000, cooldown_start=3600)
    initial = tx.init(params)
    assert (
        not initial.hyperparams_states
    )  # No hidden step schedule can override token-based values.
    zero = set_token_learning_rates(initial, config, schedule, TokenClock())
    updates, _ = tx.update(grads, zero, params)
    for update in jax.tree.leaves(updates):
        np.testing.assert_allclose(update, 0, atol=5e-7)
    peak = set_token_learning_rates(
        initial, config, schedule, TokenClock(input_tokens=100)
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
