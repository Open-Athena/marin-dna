"""PyTorch Lightning module for matched MNTP and causal-continuation arms."""

from __future__ import annotations

from typing import Any, Literal

import lightning as L
import torch
from torch import nn
from transformers import PreTrainedModel

from exp479_mntp.config import (
    COOLDOWN_START_STEP,
    TRAIN_STEPS,
    WARMUP_STEPS,
    optimizer_hyperparameters,
)
from exp479_mntp.loss import LossMetrics, per_sequence_weighted_loss
from exp479_mntp.modeling import model_logits
from exp479_mntp.optimizer import build_optimizer, build_wsd_scheduler

Arm = Literal["transferred_mntp", "scratch_mntp", "clm_continuation"]


class AdaptationModule(L.LightningModule):
    """Train one registered exp479 arm with explicit attention semantics."""

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        arm: Arm,
        batch_size: int,
        train_steps: int = TRAIN_STEPS,
        warmup_steps: int = WARMUP_STEPS,
        cooldown_start_step: int = COOLDOWN_START_STEP,
        record_gradient_norms: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.arm = arm
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.cooldown_start_step = cooldown_start_step
        self.record_gradient_norms = record_gradient_norms
        self.optimizer_values = optimizer_hyperparameters(batch_size, train_steps)
        self.gradient_norm_trace: list[dict[str, float | int]] = []
        self._latest_train_loss: float | None = None
        self.save_hyperparameters(ignore=["model"])

    @property
    def attention_mode(self) -> Literal["causal", "full"]:
        return "causal" if self.arm == "clm_continuation" else "full"

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return model_logits(
            self.model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            attention_mode=self.attention_mode,
        )

    def _metrics(self, batch: dict[str, Any]) -> LossMetrics:
        return per_sequence_weighted_loss(
            self(batch),
            batch["labels"],
            batch["loss_weights"],
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        metrics = self._metrics(batch)
        objective_loss = metrics.pooled_loss if self.arm == "clm_continuation" else metrics.loss
        if self.record_gradient_norms:
            self._latest_train_loss = float(objective_loss.detach())
        self.log(
            "train/loss",
            objective_loss,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/sequence_loss",
            metrics.loss,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/pooled_loss",
            metrics.pooled_loss,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/accuracy",
            metrics.accuracy,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        return objective_loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = self(batch)
        metrics = per_sequence_weighted_loss(
            logits,
            batch["labels"],
            batch["loss_weights"],
            z_loss_weight=0,
        )
        validation_loss = metrics.pooled_loss if self.arm == "clm_continuation" else metrics.loss
        slice_name = "diffusion" if dataloader_idx == 0 else "single_mask"
        self.log(
            f"val/{slice_name}/loss",
            validation_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["sample_ids"]),
            add_dataloader_idx=False,
        )
        self.log(
            f"val/{slice_name}/accuracy",
            metrics.accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["sample_ids"]),
            add_dataloader_idx=False,
        )

        for component in sorted(set(batch["components"])):
            in_component = torch.tensor(
                [value == component for value in batch["components"]],
                device=logits.device,
                dtype=torch.bool,
            )
            component_metrics = per_sequence_weighted_loss(
                logits[in_component],
                batch["labels"][in_component],
                batch["loss_weights"][in_component],
                z_loss_weight=0,
            )
            component_loss = (
                component_metrics.pooled_loss
                if self.arm == "clm_continuation"
                else component_metrics.loss
            )
            self.log(
                f"val/{slice_name}/component/{component}/loss",
                component_loss,
                on_step=False,
                on_epoch=True,
                batch_size=int(in_component.sum()),
                add_dataloader_idx=False,
            )
            self.log(
                f"val/{slice_name}/component/{component}/accuracy",
                component_metrics.accuracy,
                on_step=False,
                on_epoch=True,
                batch_size=int(in_component.sum()),
                add_dataloader_idx=False,
            )

        if dataloader_idx == 0 and self.arm != "clm_continuation":
            probabilities = batch["mask_probabilities"]
            for bin_index, (low, high) in enumerate(
                ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
            ):
                in_bin = (probabilities >= low) & (
                    probabilities < high if high < 1.0 else probabilities <= high
                )
                if torch.any(in_bin):
                    bin_metrics = per_sequence_weighted_loss(
                        logits[in_bin],
                        batch["labels"][in_bin],
                        batch["loss_weights"][in_bin],
                        z_loss_weight=0,
                    )
                    self.log(
                        f"val/diffusion/mask_bin_{bin_index}/loss",
                        bin_metrics.loss,
                        on_step=False,
                        on_epoch=True,
                        batch_size=int(in_bin.sum()),
                    )
                    self.log(
                        f"val/diffusion/mask_bin_{bin_index}/accuracy",
                        bin_metrics.accuracy,
                        on_step=False,
                        on_epoch=True,
                        batch_size=int(in_bin.sum()),
                    )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = build_optimizer(self.model, self.optimizer_values)
        scheduler = build_wsd_scheduler(
            optimizer,
            warmup_steps=self.warmup_steps,
            cooldown_start_step=self.cooldown_start_step,
            total_steps=self.train_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        del gradient_clip_val, gradient_clip_algorithm
        total_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.optimizer_values.max_grad_norm,
            error_if_nonfinite=self.record_gradient_norms,
        )
        if self.record_gradient_norms:
            if self._latest_train_loss is None:
                raise RuntimeError("gradient trace lacks the corresponding training loss")
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            if len(learning_rates) != 2:
                raise RuntimeError(f"expected two optimizer groups, found {len(learning_rates)}")
            norm = float(total_norm)
            self.gradient_norm_trace.append(
                {
                    "step": int(self.global_step),
                    "train_loss": self._latest_train_loss,
                    "pre_clip_gradient_norm": norm,
                    "clipped": int(norm > self.optimizer_values.max_grad_norm),
                    "adamh_learning_rate": learning_rates[0],
                    "adam_learning_rate": learning_rates[1],
                }
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["exp479"] = {
            "arm": self.arm,
            "batch_size": self.batch_size,
            "attention_mode": self.attention_mode,
            "optimizer": self.optimizer_values.to_dict(),
            "next_sequence_plan_batch": int(self.global_step),
        }
