from __future__ import annotations

import math

import pytest

from exp479_mntp.config import optimizer_hyperparameters, wsd_multiplier


@pytest.mark.parametrize(
    ("step", "expected"),
    ((0, 0.0), (50, 0.5), (100, 1.0), (799, 1.0), (800, 1.0), (900, 0.5), (1000, 0.0), (1100, 0.0)),
)
def test_wsd_boundaries(step: int, expected: float) -> None:
    assert wsd_multiplier(step) == expected


def test_optimizer_hyperparameters_apply_pinned_heuristic() -> None:
    values = optimizer_hyperparameters(batch_size=32)
    tokens = 1_000 * 32 * 256
    ratio = (32 * 2_500_000_000) / (16_384 * tokens)
    assert values.model_tokens == tokens
    assert values.nucleotide_bases == 1_000 * 32 * 255
    assert values.adamh_learning_rate == pytest.approx(
        0.015566099981405093 * math.sqrt(32 / 16_384) * (2_500_000_000 / tokens) ** 0.3
    )
    assert values.adam_learning_rate == pytest.approx(0.02989514059663958 * math.sqrt(ratio))
    assert values.epsilon == pytest.approx(1e-15 * math.sqrt(1 / ratio))
    assert values.beta2 == pytest.approx(0.9067269880630742 ** (32 / 16_384))


def test_invalid_schedule_rejected() -> None:
    with pytest.raises(ValueError, match="expected"):
        wsd_multiplier(1, warmup_steps=10, cooldown_start_step=10, total_steps=10)
