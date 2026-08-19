"""Per-sequence MNTP and weighted causal loss reducers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from exp479_mntp.masking import IGNORE_INDEX


@dataclass(frozen=True)
class LossMetrics:
    """Decision-relevant batch metrics."""

    loss: torch.Tensor
    pooled_loss: torch.Tensor
    accuracy: torch.Tensor
    selected_tokens: torch.Tensor


def per_sequence_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
) -> LossMetrics:
    """Average weighted CE within each sequence, then average sequences.

    The per-sequence denominator is the selected-token count, not the sum of
    soft-mask weights, matching the registered issue #479 objective.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must be rank 3, got shape {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2] or loss_weights.shape != labels.shape:
        raise ValueError("labels and loss_weights must match the first two logits dimensions")

    selected = labels != IGNORE_INDEX
    counts = selected.sum(dim=1)
    if torch.any(counts == 0):
        bad = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"every sequence must have at least one target; empty rows: {bad}")

    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(labels)
    weighted_loss = flat_loss * loss_weights
    sequence_losses = weighted_loss.sum(dim=1) / counts
    pooled_loss = weighted_loss.sum() / counts.sum()

    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels) & selected
    accuracy = correct.sum() / counts.sum()
    return LossMetrics(
        loss=sequence_losses.mean(),
        pooled_loss=pooled_loss,
        accuracy=accuracy,
        selected_tokens=counts.sum(),
    )


def causal_supervision(
    input_ids: torch.Tensor,
    lowercase_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align next-token labels and source case weights to output positions."""

    if input_ids.shape != lowercase_mask.shape:
        raise ValueError("lowercase_mask must have the same shape as input_ids")
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    labels[:, :-1] = input_ids[:, 1:]
    weights[:, :-1] = torch.where(lowercase_mask[:, 1:], 0.01, 1.0)
    return labels, weights
