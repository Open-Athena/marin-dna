"""Issue-specific Lightning module with exact fork and resume metadata."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from glm_experiments.exp515.config import (
    BETAS,
    BRIDGE_STEPS,
    EFFECTIVE_BATCH_SIZE,
    END_LEARNING_RATE,
    EPSILON,
    PEAK_LEARNING_RATE,
    WEIGHT_DECAY,
)
from glm_experiments.models.components.lm import HFCLM
from glm_experiments.models.components.selection import SelectorMode


def learning_rate_factor(step: int, continuation_steps: int) -> float:
    """Warm through the bridge, then cosine-decay to one tenth of peak."""

    if continuation_steps <= 0:
        raise ValueError("continuation_steps must be positive")
    if step < BRIDGE_STEPS:
        return (step + 1) / BRIDGE_STEPS
    progress = min(1.0, (step - BRIDGE_STEPS) / continuation_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        END_LEARNING_RATE / PEAK_LEARNING_RATE
        + (1.0 - END_LEARNING_RATE / PEAK_LEARNING_RATE) * cosine
    )


class Exp515Module(LightningModule):
    """HF Qwen3 CLM trained through the registered token selector."""

    def __init__(
        self,
        net: HFCLM,
        *,
        continuation_steps: int,
        plan_sha256: str,
        selector_mode: SelectorMode,
        selector_ratio: float,
        effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.continuation_steps = continuation_steps
        self.plan_sha256 = plan_sha256
        self.selector_mode = selector_mode
        self.selector_ratio = selector_ratio
        self.effective_batch_size = effective_batch_size
        self.last_selector_diagnostics: dict[str, torch.Tensor] | None = None
        self.zero_eligible_batches = 0
        self.gradient_clip_events = 0
        self.optimizer_steps_seen = 0
        self.selected_target_tokens = 0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute selection and loss with no extra diagnostic forward."""

        return self.net(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            soft_masked=batch["soft_masked"],
            soft_masked_weight=0.0,
            attention_mask=batch["attention_mask"],
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor | None:
        """Train on selected targets or skip a globally empty batch."""

        del batch_idx
        result = self.forward(batch)
        if not bool(torch.isfinite(result["loss"]).detach()):
            raise FloatingPointError("issue #515 training loss is non-finite")
        selected_count = int(result["selected_count"].detach())
        eligible_count = int(result["eligible_count"].detach())
        if selected_count == 0:
            self.zero_eligible_batches += 1
            self.last_selector_diagnostics = None
            self.log("train/zero_eligible_batch", 1.0, on_step=True)
            return None
        self.log("train/zero_eligible_batch", 0.0, on_step=True)
        self.selected_target_tokens += selected_count
        metrics = {
            "train/loss": result["loss"],
            "train/loss_full": result["loss_full"],
            "train/loss_eligible": result["loss_non_soft_masked"],
            "train/loss_selected": result["loss_selected"],
            "train/loss_unselected": result["loss_unselected"],
            "train/eligible_count": float(eligible_count),
            "train/selected_count": float(selected_count),
            "train/unselected_count": float(result["unselected_count"]),
            "train/selection_rate": selected_count / eligible_count,
            "train/processed_input_tokens": float(
                (self.global_step + 1)
                * self.effective_batch_size
                * batch["input_ids"].shape[1]
            ),
            "train/selected_target_tokens_cumulative": float(
                self.selected_target_tokens
            ),
        }
        thresholds = result["selection_thresholds"]
        finite_lower = thresholds[:, 0][torch.isfinite(thresholds[:, 0])]
        finite_upper = thresholds[:, 1][torch.isfinite(thresholds[:, 1])]
        if finite_lower.numel():
            metrics["train/selection_threshold_lower"] = finite_lower.mean()
            metrics["train/selection_threshold_upper"] = finite_upper.mean()
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)
        self.last_selector_diagnostics = {
            "eligible_mask": result["eligible_mask"].detach(),
            "selected_mask": result["selected_mask"].detach(),
            "selection_thresholds": thresholds.detach(),
            "input_ids": batch["input_ids"].detach(),
            "soft_masked": batch["soft_masked"].detach(),
        }
        return result["loss"]

    def configure_optimizers(self) -> dict[str, Any]:
        """Create the registered fresh AdamW state and shared schedule."""

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=PEAK_LEARNING_RATE,
            betas=BETAS,
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: learning_rate_factor(step, self.continuation_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Record the pre-clip norm and clipping frequency."""

        del optimizer
        total = grad_norm(self, norm_type=2)["grad_2.0_norm_total"]
        if not bool(torch.isfinite(total).detach()):
            raise FloatingPointError("issue #515 gradient norm is non-finite")
        clipped = float(total) > 1.0
        self.gradient_clip_events += int(clipped)
        self.optimizer_steps_seen += 1
        self.log("train/grad_norm_preclip", total, on_step=True)
        self.log("train/gradient_clipped", float(clipped), on_step=True)
        self.log(
            "train/gradient_clip_fraction",
            self.gradient_clip_events / self.optimizer_steps_seen,
            on_step=True,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Pin the exact data position and global RNG state into every checkpoint."""

        checkpoint["exp515"] = {
            "plan_sha256": self.plan_sha256,
            "next_sample_id": self.global_step * self.effective_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "selector_mode": self.selector_mode,
            "selector_ratio": self.selector_ratio,
            "zero_eligible_batches": self.zero_eligible_batches,
            "gradient_clip_events": self.gradient_clip_events,
            "optimizer_steps_seen": self.optimizer_steps_seen,
            "selected_target_tokens": self.selected_target_tokens,
        }
        checkpoint["exp515_rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Validate plan identity and restore every global RNG stream."""

        metadata = checkpoint.get("exp515")
        if not isinstance(metadata, dict):
            raise TypeError("checkpoint lacks exp515 resume metadata")
        if metadata["plan_sha256"] != self.plan_sha256:
            raise ValueError("checkpoint sequence plan does not match this run")
        if int(metadata["effective_batch_size"]) != self.effective_batch_size:
            raise ValueError("checkpoint effective batch does not match this run")
        self.zero_eligible_batches = int(metadata["zero_eligible_batches"])
        self.gradient_clip_events = int(metadata["gradient_clip_events"])
        self.optimizer_steps_seen = int(metadata["optimizer_steps_seen"])
        self.selected_target_tokens = int(metadata.get("selected_target_tokens", 0))
        rng = checkpoint.get("exp515_rng")
        if not isinstance(rng, dict):
            raise TypeError("checkpoint lacks exp515 RNG state")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])


def checkpoint_next_sample_id(path: str) -> int:
    """Read and validate the exact next plan row in a full checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("exp515")
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint lacks exp515 metadata")
    return int(metadata["next_sample_id"])
