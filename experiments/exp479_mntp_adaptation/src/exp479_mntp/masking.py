"""Deterministic diffusion-style MNTP corruption and shifted targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE_INDEX = -100
SOFT_MASKED_WEIGHT = 0.01


@dataclass(frozen=True)
class CorruptedBatch:
    """MNTP inputs and output-position-aligned supervision."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_weights: torch.Tensor
    target_mask: torch.Tensor
    mask_probabilities: torch.Tensor


def _splitmix64(value: int) -> int:
    """Mix an integer into a stable positive 63-bit PyTorch RNG seed."""

    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def sample_seed(base_seed: int, sample_id: int, stream: int = 0) -> int:
    """Derive one stable seed from experiment, sample, and corruption-stream IDs."""

    return _splitmix64(base_seed ^ _splitmix64(sample_id) ^ _splitmix64(stream))


def _canonical_mask(input_ids: torch.Tensor, canonical_token_ids: tuple[int, ...]) -> torch.Tensor:
    eligible = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in canonical_token_ids:
        eligible |= input_ids == token_id
    eligible[:, 0] = False
    return eligible


def corrupt_for_mntp(
    input_ids: torch.Tensor,
    lowercase_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    *,
    mask_token_id: int,
    canonical_token_ids: tuple[int, ...],
    seed: int,
) -> CorruptedBatch:
    """Mask eligible bases with one resampled Uniform(0, 1) rate per sequence.

    Labels and weights are shifted left: a selected target at input position ``i``
    supervises output position ``i - 1``. Sampling is stateless per sample ID, so
    DataLoader worker count, prefetch, and checkpoint resumption cannot change it.
    """

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank 2, got shape {tuple(input_ids.shape)}")
    if input_ids.shape != lowercase_mask.shape:
        raise ValueError("lowercase_mask must have the same shape as input_ids")
    if sample_ids.ndim != 1 or sample_ids.shape[0] != input_ids.shape[0]:
        raise ValueError("sample_ids must contain one ID per sequence")
    if input_ids.device.type != "cpu":
        raise ValueError("corruption runs on CPU before device transfer")

    eligible = _canonical_mask(input_ids, canonical_token_ids)
    if not torch.all(eligible.any(dim=1)):
        bad = (~eligible.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"sequences without an eligible A/C/G/T target: {bad}")

    corrupted = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    probabilities = torch.empty(input_ids.shape[0], dtype=torch.float32)

    for row, raw_sample_id in enumerate(sample_ids.tolist()):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed(seed, int(raw_sample_id)))
        selected = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        for _ in range(1_024):
            probability = torch.rand((), generator=generator)
            selected = torch.rand(input_ids.shape[1], generator=generator) < probability
            selected &= eligible[row]
            if selected.any():
                break
        else:
            raise RuntimeError(f"failed to sample a target for sample {raw_sample_id}")

        target_positions = selected.nonzero(as_tuple=False).flatten()
        output_positions = target_positions - 1
        probabilities[row] = probability
        labels[row, output_positions] = input_ids[row, target_positions]
        loss_weights[row, output_positions] = torch.where(
            lowercase_mask[row, target_positions],
            SOFT_MASKED_WEIGHT,
            1.0,
        )
        target_mask[row, output_positions] = True
        corrupted[row, target_positions] = mask_token_id

    return CorruptedBatch(
        input_ids=corrupted,
        labels=labels,
        loss_weights=loss_weights,
        target_mask=target_mask,
        mask_probabilities=probabilities,
    )


def corrupt_single_mask(
    input_ids: torch.Tensor,
    lowercase_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    *,
    mask_token_id: int,
    canonical_token_ids: tuple[int, ...],
    seed: int,
) -> CorruptedBatch:
    """Select exactly one deterministic eligible target per sequence."""

    eligible = _canonical_mask(input_ids, canonical_token_ids)
    corrupted = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for row, raw_sample_id in enumerate(sample_ids.tolist()):
        positions = eligible[row].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError(f"sample {raw_sample_id} has no eligible A/C/G/T target")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed(seed, int(raw_sample_id), stream=1))
        selected_index = int(torch.randint(positions.numel(), (), generator=generator))
        target_position = positions[selected_index]
        output_position = target_position - 1
        labels[row, output_position] = input_ids[row, target_position]
        loss_weights[row, output_position] = (
            SOFT_MASKED_WEIGHT if lowercase_mask[row, target_position] else 1.0
        )
        target_mask[row, output_position] = True
        corrupted[row, target_position] = mask_token_id

    return CorruptedBatch(
        input_ids=corrupted,
        labels=labels,
        loss_weights=loss_weights,
        target_mask=target_mask,
        mask_probabilities=torch.full((input_ids.shape[0],), 1.0 / input_ids.shape[1]),
    )
