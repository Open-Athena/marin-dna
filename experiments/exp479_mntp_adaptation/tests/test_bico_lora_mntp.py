from __future__ import annotations

import pytest
import torch

from exp479_mntp.bico_lora_mntp import (
    BICO_LORA_REFERENCE_RUNTIME_SECONDS,
    BICO_LORA_STANDARD_LEARNING_RATE,
    BicoLoraConfig,
    excluded_mntp_key_mask,
    projected_bico_base_run_hours,
)


def test_bico_lora_config_uses_one_maximal_physical_batch_without_accumulation() -> None:
    config = BicoLoraConfig(batch_size=137)
    assert config.batch_size == 137
    assert config.microbatch_size == 137
    assert config.accumulation_steps == 1
    assert config.rank == 16
    assert config.learning_rate == 1e-5
    assert config.warmup_steps == 100
    assert config.cooldown_start_step == 800
    assert config.train_steps == 1_000


def test_bico_lora_standard_rate_changes_only_the_registered_peak_rate() -> None:
    conservative = BicoLoraConfig(batch_size=94)
    standard = BicoLoraConfig(batch_size=94, learning_rate=BICO_LORA_STANDARD_LEARNING_RATE)
    conservative_payload = conservative.to_dict()
    standard_payload = standard.to_dict()
    assert standard.learning_rate == 5e-5
    assert standard.accumulation_steps == 1
    assert standard.batch_size == 94
    assert {
        key: value for key, value in conservative_payload.items() if key != "learning_rate"
    } == {key: value for key, value in standard_payload.items() if key != "learning_rate"}


def test_batch_94_budget_uses_observed_sustained_runtime_not_cold_steps() -> None:
    hours, basis = projected_bico_base_run_hours(
        batch_size=94,
        cold_seconds_per_step=8.9179584675,
    )

    assert hours == pytest.approx(BICO_LORA_REFERENCE_RUNTIME_SECONDS / 3_600)
    assert "t37n0upf" in basis


def test_unknown_batch_budget_falls_back_to_cold_step_extrapolation() -> None:
    hours, basis = projected_bico_base_run_hours(
        batch_size=93,
        cold_seconds_per_step=4.0,
    )

    assert hours == pytest.approx(4_000 / 3_600)
    assert basis == "two-cold-step extrapolation"
    with pytest.raises(ValueError, match="positive"):
        projected_bico_base_run_hours(batch_size=94, cold_seconds_per_step=0)


def test_excluded_mntp_key_mask_excludes_every_shifted_pad_target() -> None:
    pad_token_id = 7
    input_ids = torch.tensor([[2, pad_token_id, 3, pad_token_id, 4]])
    labels = torch.tensor([[11, -100, 13, -100, -100]])
    token_mask = torch.tensor([[1, 1, 1, 1, 0]])
    mask = excluded_mntp_key_mask(
        token_mask,
        labels,
        input_ids,
        pad_token_id=pad_token_id,
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 5, 5)
    assert torch.all(mask[0, 0, :, [1, 3, 4]] < -1e30)
    assert torch.all(mask[0, 0, :, [0, 2]] == 0)


def test_excluded_mntp_key_mask_rejects_unshifted_or_non_pad_target() -> None:
    with pytest.raises(RuntimeError, match="not the registered PAD"):
        excluded_mntp_key_mask(
            torch.ones((1, 4), dtype=torch.long),
            torch.tensor([[9, -100, -100, -100]]),
            torch.tensor([[2, 3, 4, 5]]),
            pad_token_id=7,
            dtype=torch.float32,
        )
