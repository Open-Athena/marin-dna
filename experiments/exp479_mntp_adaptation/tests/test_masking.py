from __future__ import annotations

import torch

from exp479_mntp.loss import per_sequence_weighted_loss
from exp479_mntp.masking import IGNORE_INDEX, corrupt_for_mntp, corrupt_single_mask


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [2, 3, 4, 5, 6, 0, 1],
            [2, 6, 5, 4, 3, 0, 1],
        ],
        dtype=torch.long,
    )
    lowercase = torch.tensor(
        [
            [False, False, True, False, True, False, False],
            [False, True, False, True, False, False, False],
        ]
    )
    sample_ids = torch.tensor([10, 11])
    return input_ids, lowercase, sample_ids


def test_corruption_is_deterministic_and_excludes_special_tokens() -> None:
    input_ids, lowercase, sample_ids = _inputs()
    first = corrupt_for_mntp(
        input_ids,
        lowercase,
        sample_ids,
        mask_token_id=7,
        canonical_token_ids=(3, 4, 5, 6),
        seed=42,
    )
    second = corrupt_for_mntp(
        input_ids,
        lowercase,
        sample_ids,
        mask_token_id=7,
        canonical_token_ids=(3, 4, 5, 6),
        seed=42,
    )
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.loss_weights, second.loss_weights)
    assert torch.equal(first.input_ids[:, 0], input_ids[:, 0])
    assert torch.equal(first.input_ids[:, 5:], input_ids[:, 5:])
    assert torch.all((first.labels != IGNORE_INDEX).sum(dim=1) >= 1)


def test_labels_are_shifted_and_selected_targets_are_removed() -> None:
    input_ids, lowercase, sample_ids = _inputs()
    result = corrupt_for_mntp(
        input_ids,
        lowercase,
        sample_ids,
        mask_token_id=7,
        canonical_token_ids=(3, 4, 5, 6),
        seed=9,
    )
    for row in range(input_ids.shape[0]):
        output_positions = (result.labels[row] != IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        for output_position in output_positions.tolist():
            target_position = output_position + 1
            assert result.labels[row, output_position] == input_ids[row, target_position]
            assert result.input_ids[row, target_position] == 7
            expected_weight = 0.01 if lowercase[row, target_position] else 1.0
            assert torch.isclose(
                result.loss_weights[row, output_position], torch.tensor(expected_weight)
            )


def test_single_mask_selects_exactly_one_target() -> None:
    input_ids, lowercase, sample_ids = _inputs()
    result = corrupt_single_mask(
        input_ids,
        lowercase,
        sample_ids,
        mask_token_id=7,
        canonical_token_ids=(3, 4, 5, 6),
        seed=123,
    )
    assert torch.equal((result.labels != IGNORE_INDEX).sum(dim=1), torch.ones(2, dtype=torch.long))
    assert torch.equal((result.input_ids == 7).sum(dim=1), torch.ones(2, dtype=torch.long))


def test_sequence_loss_does_not_overweight_high_mask_rows() -> None:
    logits = torch.zeros((2, 4, 3))
    labels = torch.tensor([[0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX], [1, 1, 1, IGNORE_INDEX]])
    weights = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    logits[0, 0, 0] = 4.0
    logits[1, :3, 1] = -4.0
    metrics = per_sequence_weighted_loss(logits, labels, weights, z_loss_weight=0)
    row0 = torch.nn.functional.cross_entropy(logits[0, :1], labels[0, :1])
    row1 = torch.nn.functional.cross_entropy(logits[1, :3], labels[1, :3])
    assert metrics.loss == torch.mean(torch.stack((row0, row1)))
    assert metrics.pooled_loss != metrics.loss


def test_repeat_weights_normalize_by_weight_sum() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0]],
            [[0.0, 3.0], [3.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, 0], [0, 0]])
    weights = torch.tensor([[1.0, 0.01], [1.0, 1.0]])
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)

    metrics = per_sequence_weighted_loss(logits, labels, weights, z_loss_weight=0)
    expected_rows = (token_losses * weights).sum(dim=1) / weights.sum(dim=1)
    expected_pooled = (token_losses * weights).sum() / weights.sum()

    assert metrics.loss == torch.mean(expected_rows)
    assert metrics.pooled_loss == expected_pooled
    assert metrics.loss_weight_sum == weights.sum()
    assert metrics.weighted_loss_sum == (token_losses * weights).sum()
    assert metrics.pooled_loss != (token_losses * weights).sum() / labels.numel()


def test_sequence_without_canonical_target_is_rejected() -> None:
    with torch.no_grad():
        input_ids = torch.tensor([[2, 1, 0]])
        lowercase = torch.zeros_like(input_ids, dtype=torch.bool)
        sample_ids = torch.tensor([0])
    try:
        corrupt_for_mntp(
            input_ids,
            lowercase,
            sample_ids,
            mask_token_id=7,
            canonical_token_ids=(3, 4, 5, 6),
            seed=0,
        )
    except ValueError as error:
        assert "without an eligible" in str(error)
    else:
        raise AssertionError("expected zero-target input to fail")
