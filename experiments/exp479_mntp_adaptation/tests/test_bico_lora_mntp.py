from __future__ import annotations

import pytest
import torch

from exp479_mntp.bico_lora_mntp import BicoLoraConfig, excluded_mntp_key_mask


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
