"""Issue-specific Lightning module with exact fork and resume metadata."""

from __future__ import annotations

import math
import random
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from glm_experiments.exp515.config import (
    BETAS,
    BRIDGE_STEPS,
    CDS_ARM_WARMUP_STEPS,
    EFFECTIVE_BATCH_SIZE,
    END_LEARNING_RATE,
    EPSILON,
    PEAK_LEARNING_RATE,
    WEIGHT_DECAY,
    ObjectiveKind,
)
from glm_experiments.models.components.lm import HFCLM
from glm_experiments.models.components.selection import SelectorMode, select_token_mask

ScheduleKind = Literal[
    "warmup_cosine",
    "warmup_constant",
    "arm_warmup_constant",
]


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


def constant_after_warmup_learning_rate_factor(step: int) -> float:
    """Warm through the bridge, then hold the peak learning rate."""

    if step < BRIDGE_STEPS:
        return (step + 1) / BRIDGE_STEPS
    return 1.0


def arm_warmup_constant_learning_rate_factor(step: int) -> float:
    """Warm a fresh arm optimizer, then hold the exp58 learning rate."""

    if step < CDS_ARM_WARMUP_STEPS:
        return (step + 1) / CDS_ARM_WARMUP_STEPS
    return 1.0


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
        objective_kind: ObjectiveKind = "hard_ce",
        resume_plan_sha256: str | None = None,
        teacher_checkpoint: str | None = None,
        schedule_kind: ScheduleKind = "warmup_cosine",
        effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
        sample_id_offset: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.continuation_steps = continuation_steps
        self.plan_sha256 = plan_sha256
        self.resume_plan_sha256 = resume_plan_sha256
        self.selector_mode = selector_mode
        self.selector_ratio = selector_ratio
        self.objective_kind = objective_kind
        self.teacher_checkpoint = teacher_checkpoint
        self.schedule_kind = schedule_kind
        self.effective_batch_size = effective_batch_size
        self.sample_id_offset = sample_id_offset
        self.last_selector_diagnostics: dict[str, torch.Tensor] | None = None
        self.zero_eligible_batches = 0
        self.gradient_clip_events = 0
        self.optimizer_steps_seen = 0
        self.selected_target_tokens = 0
        self._teacher_net: HFCLM | None = None

        if sample_id_offset < 0:
            raise ValueError("sample ID offset must be non-negative")
        if objective_kind == "hard_ce" and teacher_checkpoint is not None:
            raise ValueError("hard-CE objective must not load a teacher")
        if objective_kind != "hard_ce" and teacher_checkpoint is None:
            raise ValueError("teacher objective requires a checkpoint")

    def on_fit_start(self) -> None:
        """Load a frozen teacher without registering it in student checkpoints."""

        if self.objective_kind == "hard_ce":
            return
        assert self.teacher_checkpoint is not None
        teacher = HFCLM(
            self.teacher_checkpoint,
            torch_dtype="bfloat16",
            selector_enabled=False,
        )
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.to(self.device)
        object.__setattr__(self, "_teacher_net", teacher)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denominator = mask.sum().clamp_min(1)
        return values.masked_select(mask).sum() / denominator

    @staticmethod
    def _teacher_selection_thresholds(
        teacher_loss: torch.Tensor,
        eligible: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        bounds = torch.full(
            (teacher_loss.shape[0], 2),
            torch.nan,
            device=teacher_loss.device,
            dtype=teacher_loss.dtype,
        )
        selected_loss = teacher_loss.masked_fill(~(selected & eligible), float("inf"))
        bounds[:, 0] = selected_loss.amin(dim=1)
        bounds[:, 1] = teacher_loss.masked_fill(
            ~(selected & eligible),
            float("-inf"),
        ).amax(dim=1)
        bounds.masked_fill_(~selected.any(dim=1, keepdim=True), torch.nan)
        return bounds

    def _teacher_result(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute pure teacher KL or hard CE on frozen-teacher low-loss targets."""

        teacher = self._teacher_net
        if teacher is None:
            raise RuntimeError("teacher was not loaded before the training step")
        student_logits = self.net.get_logits(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )[:, :-1]
        with torch.no_grad():
            teacher_logits = teacher.get_logits(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )[:, :-1]
        labels = batch["labels"][:, 1:].long()
        valid = (labels != -100) & batch["attention_mask"][:, 1:].bool()
        eligible = valid & ~batch["soft_masked"][:, 1:].bool()
        teacher_ce = F.cross_entropy(
            teacher_logits.transpose(1, 2).float(),
            labels,
            reduction="none",
            ignore_index=-100,
        )
        student_ce = F.cross_entropy(
            student_logits.transpose(1, 2).float(),
            labels,
            reduction="none",
            ignore_index=-100,
        )
        if self.objective_kind == "teacher_kl":
            teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
            teacher_prob = teacher_log_prob.exp()
            student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
            loss_per_token = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(
                dim=-1
            )
            selected = eligible
            thresholds = torch.full(
                (labels.shape[0], 2),
                torch.nan,
                device=labels.device,
                dtype=loss_per_token.dtype,
            )
        elif self.objective_kind == "teacher_low":
            loss_per_token = student_ce
            selected = select_token_mask(
                teacher_ce,
                eligible,
                mode="student_low",
                ratio=self.selector_ratio,
            )
            thresholds = self._teacher_selection_thresholds(
                teacher_ce,
                eligible,
                selected,
            )
        else:
            raise ValueError(f"unexpected objective {self.objective_kind!r}")
        unselected = eligible & ~selected
        return {
            "loss": self._masked_mean(loss_per_token, selected),
            "loss_full": self._masked_mean(loss_per_token, valid),
            "loss_non_soft_masked": self._masked_mean(loss_per_token, eligible),
            "loss_selected": self._masked_mean(loss_per_token, selected),
            "loss_unselected": self._masked_mean(loss_per_token, unselected),
            "loss_per_token": loss_per_token,
            "aligned_labels": labels,
            "eligible_mask": eligible,
            "selected_mask": selected,
            "unselected_mask": unselected,
            "selection_thresholds": thresholds,
            "eligible_count": eligible.sum(),
            "selected_count": selected.sum(),
            "unselected_count": unselected.sum(),
        }

    def release_teacher(self) -> None:
        """Release the unregistered frozen teacher before evaluation/checkpoint return."""

        object.__setattr__(self, "_teacher_net", None)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute selection and loss with no extra diagnostic forward."""

        if self.objective_kind != "hard_ce":
            return self._teacher_result(batch)
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

        if self.schedule_kind == "arm_warmup_constant":
            learning_rate_lambda = arm_warmup_constant_learning_rate_factor
        elif self.schedule_kind == "warmup_constant":
            learning_rate_lambda = constant_after_warmup_learning_rate_factor
        else:
            learning_rate_lambda = lambda step: learning_rate_factor(
                step,
                self.continuation_steps,
            )

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=PEAK_LEARNING_RATE,
            betas=BETAS,
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=learning_rate_lambda,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Record the pre-clip norm and clipping frequency."""

        del optimizer
        total = grad_norm(self.net, norm_type=2)["grad_2.0_norm_total"]
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
            "parent_plan_sha256": self.resume_plan_sha256,
            "next_sample_id": self.sample_id_offset
            + self.global_step * self.effective_batch_size,
            "sample_id_offset": self.sample_id_offset,
            "effective_batch_size": self.effective_batch_size,
            "selector_mode": self.selector_mode,
            "selector_ratio": self.selector_ratio,
            "objective_kind": self.objective_kind,
            "teacher_checkpoint": self.teacher_checkpoint,
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
        accepted_plan_sha256s = {self.plan_sha256}
        if self.resume_plan_sha256 is not None:
            accepted_plan_sha256s.add(self.resume_plan_sha256)
        if metadata["plan_sha256"] not in accepted_plan_sha256s:
            raise ValueError(
                "checkpoint sequence plan does not match this run or its "
                "validated parent plan"
            )
        if int(metadata["effective_batch_size"]) != self.effective_batch_size:
            raise ValueError("checkpoint effective batch does not match this run")
        if int(metadata.get("sample_id_offset", 0)) != self.sample_id_offset:
            raise ValueError("checkpoint sample offset does not match this run")
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
