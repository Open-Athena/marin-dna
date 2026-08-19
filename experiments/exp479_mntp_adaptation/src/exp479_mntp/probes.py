"""Fixed left/right context-dependence probes for exp479 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from exp479_mntp.config import NUCLEOTIDE_LENGTH
from exp479_mntp.modeling import model_logits


def _next_base_ids(token_ids: torch.Tensor, canonical_ids: tuple[int, ...]) -> torch.Tensor:
    replacement = token_ids.clone()
    recognized = torch.zeros_like(token_ids, dtype=torch.bool)
    for index, token_id in enumerate(canonical_ids):
        selected = token_ids == token_id
        replacement[selected] = canonical_ids[(index + 1) % len(canonical_ids)]
        recognized |= selected
    if not torch.all(recognized):
        raise ValueError("fixed context probe encountered a noncanonical flank token")
    return replacement


@torch.no_grad()
def context_dependence(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    target_input_position: int,
    mask_token_id: int | None,
    canonical_ids: tuple[int, ...],
    attention_mode: Literal["causal", "full"],
    flank_offset: int = 16,
) -> dict[str, float]:
    """Measure mean L1 target-distribution changes from fixed flank substitutions."""

    if not 0 < flank_offset < target_input_position:
        raise ValueError("flank_offset must leave a valid left-flank position")
    right_position = target_input_position + flank_offset
    if right_position >= input_ids.shape[1]:
        raise ValueError("flank_offset must leave a valid right-flank position")

    base = input_ids.clone()
    if mask_token_id is not None:
        base[:, target_input_position] = mask_token_id
    left = base.clone()
    right = base.clone()
    left[:, target_input_position - flank_offset] = _next_base_ids(
        left[:, target_input_position - flank_offset], canonical_ids
    )
    right[:, right_position] = _next_base_ids(right[:, right_position], canonical_ids)
    stacked_ids = torch.cat((base, left, right))
    stacked_mask = torch.cat((attention_mask, attention_mask, attention_mask))
    logits = model_logits(
        model,
        input_ids=stacked_ids,
        attention_mask=stacked_mask,
        attention_mode=attention_mode,
    )[:, target_input_position - 1, list(canonical_ids)]
    probabilities = torch.softmax(logits.float(), dim=-1)
    base_probabilities, left_probabilities, right_probabilities = probabilities.chunk(3)
    return {
        "left_l1": float((base_probabilities - left_probabilities).abs().sum(dim=-1).mean()),
        "right_l1": float((base_probabilities - right_probabilities).abs().sum(dim=-1).mean()),
    }


class ContextProbeCallback(L.Callback):
    """Log a fixed probe at every validation/checkpoint boundary."""

    def __init__(
        self,
        *,
        validation_plan: Path,
        tokenizer: PreTrainedTokenizerBase,
        mask_token_id: int | None,
        canonical_ids: tuple[int, ...],
        sample_count: int = 32,
    ) -> None:
        sequences: list[str] = []
        with validation_plan.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sequences.append(str(json.loads(line)["sequence"]))
                    if len(sequences) == sample_count:
                        break
        if len(sequences) != sample_count:
            raise ValueError(f"context probe needs {sample_count} validation sequences")
        if any(len(sequence) != NUCLEOTIDE_LENGTH for sequence in sequences):
            raise ValueError("context probe sequence length differs from registered context")
        encoded: dict[str, Any] = tokenizer(
            sequences,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=NUCLEOTIDE_LENGTH + 1,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].long()
        self.attention_mask = encoded["attention_mask"].long()
        self.target_input_position = 1 + NUCLEOTIDE_LENGTH // 2
        self.mask_token_id = mask_token_id
        self.canonical_ids = canonical_ids

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        device = pl_module.device
        values = context_dependence(
            pl_module.model,
            self.input_ids.to(device),
            self.attention_mask.to(device),
            target_input_position=self.target_input_position,
            mask_token_id=self.mask_token_id,
            canonical_ids=self.canonical_ids,
            attention_mode=pl_module.attention_mode,
        )
        trainer.logger.log_metrics(
            {f"probe/context_{name}": value for name, value in values.items()},
            step=trainer.global_step,
        )
