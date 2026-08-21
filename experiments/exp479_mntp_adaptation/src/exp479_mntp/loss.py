"""Per-sequence MNTP and weighted causal loss reducers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from exp479_mntp.config import SOURCE_Z_LOSS_WEIGHT
from exp479_mntp.masking import IGNORE_INDEX


@dataclass(frozen=True)
class LossMetrics:
    """Decision-relevant batch metrics."""

    loss: torch.Tensor
    pooled_loss: torch.Tensor
    accuracy: torch.Tensor
    selected_tokens: torch.Tensor
    loss_weight_sum: torch.Tensor
    weighted_loss_sum: torch.Tensor


def per_sequence_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    z_loss_weight: float = SOURCE_Z_LOSS_WEIGHT,
) -> LossMetrics:
    """Reduce weighted CE with sequence-balanced and token-weighted means.

    ``loss`` gives every sequence equal weight after normalizing by that
    sequence's effective loss-weight sum. ``pooled_loss`` matches Marin's
    weighted mean across all selected tokens in the batch.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must be rank 3, got shape {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2] or loss_weights.shape != labels.shape:
        raise ValueError("labels and loss_weights must match the first two logits dimensions")
    if z_loss_weight < 0:
        raise ValueError("z_loss_weight must be non-negative")

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
    if z_loss_weight:
        flat_loss = flat_loss + torch.logsumexp(logits.float(), dim=-1).square() * z_loss_weight
    if torch.any(loss_weights < 0):
        raise ValueError("loss_weights must be non-negative")
    effective_weights = torch.where(selected, loss_weights, 0.0)
    weight_sums = effective_weights.sum(dim=1)
    if torch.any(weight_sums <= 0):
        bad = (weight_sums <= 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"every sequence must have positive loss weight; zero rows: {bad}")

    weighted_loss = flat_loss * effective_weights
    weighted_loss_sum = weighted_loss.sum()
    loss_weight_sum = weight_sums.sum()
    sequence_losses = weighted_loss.sum(dim=1) / weight_sums
    pooled_loss = weighted_loss_sum / loss_weight_sum

    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels) & selected
    accuracy = correct.sum() / counts.sum()
    return LossMetrics(
        loss=sequence_losses.mean(),
        pooled_loss=pooled_loss,
        accuracy=accuracy,
        selected_tokens=counts.sum(),
        loss_weight_sum=loss_weight_sum,
        weighted_loss_sum=weighted_loss_sum,
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
