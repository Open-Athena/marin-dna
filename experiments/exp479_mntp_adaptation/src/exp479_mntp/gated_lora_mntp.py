"""Causal-preserving dual-path LoRA adaptation with a right-context-use gate."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import save_file
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from exp479_mntp.callbacks import BudgetGuardCallback, RuntimeMetricsCallback
from exp479_mntp.causal_longrun import (
    LONGRUN_CHECKPOINT_STEPS,
    _artifact_record,
    _write_retention_manifest,
    plot_longrun_stability,
)
from exp479_mntp.checkpoint_audit import assert_plan_contract
from exp479_mntp.config import (
    BUDGET_USD,
    EXPERIMENT_TAGS,
    MODEL_ID,
    MODEL_REVISION,
    WANDB_PROJECT,
    wsd_multiplier,
)
from exp479_mntp.data import plan_sha256
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.lora_mntp import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_EFFECTIVE_BATCH_SIZE,
    LORA_MASK_PROBABILITY,
    LORA_MICROBATCH_SIZE,
    LORA_RANK,
    LORA_TARGET_MODULES,
    LoraMntpConfig,
    _evaluate_preserving_mode,
    assert_lora_trainables,
    build_lora_bundle,
)
from exp479_mntp.loss import per_sequence_weighted_loss
from exp479_mntp.masking import IGNORE_INDEX
from exp479_mntp.modeling import ModelBundle
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    information_gate,
    paired_comparison,
    summarize_readouts,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

GATED_AUXILIARY_BRANCH_WEIGHT = 1.0
GATED_LEARNING_RATE = 1e-5
GATED_WARMUP_STEPS = 100
GATED_COOLDOWN_START_STEP = 800
GATED_TRAIN_STEPS = 1_000
GATED_MAX_GRAD_NORM = 1.0
GATED_MAX_INSTANCE_HOURS = 4.0
GATED_WANDB_GROUP = "dna-exp479-causal-preserving-bidirectional"
GATED_RUN_NAME = "dna-exp479-gated-dualpath-lora-r16-mntp-unk20pct-lr1e-5-wsd1000-seed0"
GATED_MODEL_ARTIFACT_PREFIX = "dna-exp479-gated-dualpath-lora-r16-mntp-unk"
GATED_EVALUATION_ARTIFACT = "dna-exp479-gated-dualpath-lora-information-gate"
DECTOENC_REFERENCE = "https://doi.org/10.1016/j.knosys.2024.112907"
BITUNE_REFERENCE = "https://arxiv.org/abs/2405.14862"


@dataclass(frozen=True)
class GatedLoraConfig:
    """Registered conservative configuration for a causal-preserving LoRA path."""

    rank: int = LORA_RANK
    alpha: int = LORA_ALPHA
    dropout: float = LORA_DROPOUT
    mask_probability: float = LORA_MASK_PROBABILITY
    auxiliary_branch_loss_weight: float = GATED_AUXILIARY_BRANCH_WEIGHT
    learning_rate: float = GATED_LEARNING_RATE
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = GATED_MAX_GRAD_NORM
    warmup_steps: int = GATED_WARMUP_STEPS
    cooldown_start_step: int = GATED_COOLDOWN_START_STEP
    train_steps: int = GATED_TRAIN_STEPS
    microbatch_size: int = LORA_MICROBATCH_SIZE
    accumulation_steps: int = LORA_EFFECTIVE_BATCH_SIZE // LORA_MICROBATCH_SIZE

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not 0 < self.mask_probability <= 1:
            raise ValueError("MNTP mask probability must be in (0, 1]")
        if self.auxiliary_branch_loss_weight <= 0:
            raise ValueError("auxiliary branch loss weight must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if self.microbatch_size * self.accumulation_steps != LORA_EFFECTIVE_BATCH_SIZE:
            raise ValueError("microbatch and accumulation must preserve effective batch 64")
        wsd_multiplier(
            0,
            warmup_steps=self.warmup_steps,
            cooldown_start_step=self.cooldown_start_step,
            total_steps=self.train_steps,
        )

    def to_dict(self) -> dict[str, float | int | str | list[str]]:
        """Return the complete registered configuration."""

        return asdict(self) | {
            "target_modules": list(LORA_TARGET_MODULES),
            "attention_schedule": "causal source path plus full-attention LoRA branch",
            "mixture": "causal_logits + tanh(causal_logits @ zero_vector) * branch_delta",
        }


def configure_gated_numerics() -> dict[str, str | bool]:
    """Match the numeric state used by the successful serialization audit."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    return {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


class FloatLogitView(nn.Module):
    """Expose one model with float32 logits for exact paired arithmetic."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        outputs = self.model(**kwargs)
        return SimpleNamespace(logits=outputs.logits.float())


class CausalPreservingGatedModel(nn.Module):
    """Mix a frozen causal pass with a separately adapted bidirectional pass."""

    def __init__(self, peft_model: PeftModel, *, vocab_size: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("gate vocabulary size must be positive")
        self.peft_model = peft_model
        self.mixing_gate = nn.Linear(vocab_size, 1, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.mixing_gate.weight)

    def forward_components(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        branch_is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixed, source, branch logits and token-wise gate coefficients."""

        with torch.no_grad(), self.peft_model.disable_adapter():
            causal_outputs = self.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                is_causal=True,
                return_dict=True,
            )
        causal_logits = causal_outputs.logits.detach().float()
        branch_outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            is_causal=branch_is_causal,
            return_dict=True,
        )
        branch_logits = branch_outputs.logits.float()
        coefficients = torch.tanh(self.mixing_gate(causal_logits))
        mixed_logits = causal_logits + coefficients * (branch_logits - causal_logits)
        return mixed_logits, causal_logits, branch_logits, coefficients

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del kwargs
        if input_ids is None or attention_mask is None:
            raise ValueError("gated model requires input_ids and attention_mask")
        mixed_logits, _, _, _ = self.forward_components(
            input_ids=input_ids,
            attention_mask=attention_mask,
            branch_is_causal=is_causal,
        )
        return SimpleNamespace(logits=mixed_logits)


def assert_gated_trainables(model: CausalPreservingGatedModel) -> dict[str, int]:
    """Require only LoRA matrices and the seven-value mixing vector to be trainable."""

    lora_count, _ = assert_lora_trainables(model.peft_model)
    trainable = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    unexpected = [
        name
        for name, _ in trainable
        if ".lora_A." not in name and ".lora_B." not in name and name != "mixing_gate.weight"
    ]
    if unexpected:
        raise RuntimeError(f"unexpected gated trainable parameters: {unexpected[:5]}")
    gate_count = model.mixing_gate.weight.numel()
    if gate_count != model.mixing_gate.in_features:
        raise RuntimeError("mixing gate is not one scalar projection over source logits")
    if not model.mixing_gate.weight.requires_grad:
        raise RuntimeError("mixing gate is frozen")
    total = sum(parameter.numel() for _, parameter in trainable)
    if total != lora_count + gate_count:
        raise RuntimeError("gated trainable parameter count is inconsistent")
    return {"lora": lora_count, "gate": gate_count, "total": total}


def right_context_use_gate(comparison: dict[str, object]) -> dict[str, object]:
    """Require right context to be non-inferior on both metrics and strictly useful on one."""

    ce_delta = float(comparison["nucleotide_ce_delta"])
    accuracy_delta = float(comparison["nucleotide_accuracy_delta"])
    ce_ci_high = float(comparison["nucleotide_ce_delta_ci95_high"])
    accuracy_ci_low = float(comparison["nucleotide_accuracy_delta_ci95_low"])
    point_noninferior = ce_delta <= 0 and accuracy_delta >= 0
    confidence_noninferior = ce_ci_high <= 0 and accuracy_ci_low >= 0
    point_strict = ce_delta < 0 or accuracy_delta > 0
    confidence_strict = ce_ci_high < 0 or accuracy_ci_low > 0
    return {
        "candidate": comparison["candidate"],
        "baseline": comparison["baseline"],
        "criterion": (
            "full branch is non-inferior to its causalized ablation on both paired metrics "
            "and strictly improves at least one with 95% support"
        ),
        "point_noninferior": point_noninferior,
        "confidence_noninferior": confidence_noninferior,
        "point_strict_improvement": point_strict,
        "confidence_strict_improvement": confidence_strict,
        "passed": point_noninferior
        and confidence_noninferior
        and point_strict
        and confidence_strict,
    }


class GatedLoraModule(L.LightningModule):
    """Train the LoRA branch and gate while leaving the causal source path frozen."""

    def __init__(self, *, model: CausalPreservingGatedModel, config: GatedLoraConfig) -> None:
        super().__init__()
        self.model = model
        self.gated_config = config
        self.gradient_norm_trace: list[dict[str, float | int]] = []
        self._latest_trace: dict[str, float] | None = None
        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        mixed, _, branch, coefficients = self.model.forward_components(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            branch_is_causal=False,
        )
        mixed_metrics = per_sequence_weighted_loss(
            mixed,
            batch["labels"],
            batch["loss_weights"],
        )
        branch_metrics = per_sequence_weighted_loss(
            branch,
            batch["labels"],
            batch["loss_weights"],
        )
        total_loss = (
            mixed_metrics.loss
            + self.gated_config.auxiliary_branch_loss_weight * branch_metrics.loss
        )
        selected = batch["labels"] != IGNORE_INDEX
        selected_coefficients = coefficients.squeeze(-1)[selected]
        self._latest_trace = {
            "train_loss": float(total_loss.detach()),
            "gated_loss": float(mixed_metrics.loss.detach()),
            "branch_loss": float(branch_metrics.loss.detach()),
            "gated_accuracy": float(mixed_metrics.accuracy.detach()),
            "branch_accuracy": float(branch_metrics.accuracy.detach()),
            "gate_mean": float(selected_coefficients.mean().detach()),
            "gate_mean_absolute": float(selected_coefficients.abs().mean().detach()),
            "gate_maximum_absolute": float(selected_coefficients.abs().max().detach()),
        }
        for name, value in self._latest_trace.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                batch_size=self.gated_config.microbatch_size,
            )
        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        assert_gated_trainables(self.model)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.gated_config.learning_rate,
            betas=(self.gated_config.beta1, self.gated_config.beta2),
            eps=self.gated_config.epsilon,
            weight_decay=self.gated_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: wsd_multiplier(
                step,
                warmup_steps=self.gated_config.warmup_steps,
                cooldown_start_step=self.gated_config.cooldown_start_step,
                total_steps=self.gated_config.train_steps,
            ),
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
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        gate_gradient = self.model.mixing_gate.weight.grad
        gate_norm = 0.0 if gate_gradient is None else float(gate_gradient.detach().float().norm())
        total_norm_tensor = nn.utils.clip_grad_norm_(
            parameters,
            self.gated_config.max_grad_norm,
            error_if_nonfinite=True,
        )
        total_norm = float(total_norm_tensor)
        lora_norm = float(max(total_norm**2 - gate_norm**2, 0.0) ** 0.5)
        if self._latest_trace is None:
            raise RuntimeError("gated gradient trace lacks its corresponding training forward")
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        if len(learning_rates) != 1:
            raise RuntimeError(f"expected one gated optimizer group, found {len(learning_rates)}")
        row: dict[str, float | int] = {
            "step": int(self.global_step),
            **self._latest_trace,
            "pre_clip_gradient_norm": total_norm,
            "lora_gradient_norm": lora_norm,
            "gate_gradient_norm": gate_norm,
            "clipped": int(total_norm > self.gated_config.max_grad_norm),
            "learning_rate": learning_rates[0],
            "gate_weight_norm": float(self.model.mixing_gate.weight.detach().float().norm()),
        }
        self.gradient_norm_trace.append(row)
        for name in (
            "pre_clip_gradient_norm",
            "lora_gradient_norm",
            "gate_gradient_norm",
            "clipped",
            "learning_rate",
            "gate_weight_norm",
        ):
            self.log(
                f"train/{name}",
                row[name],
                on_step=True,
                on_epoch=False,
                batch_size=self.gated_config.microbatch_size,
            )


def _same_paired_scores(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = (
        "sample_id",
        "target_nucleotide_index",
        "nucleotide_ce",
        "nucleotide_correct",
        "full_vocab_ce",
        "full_vocab_correct",
    )
    return all(
        np.array_equal(left[column].to_numpy(), right[column].to_numpy()) for column in columns
    )


class RetainedGatedTrajectoryCallback(L.Callback):
    """Evaluate the gated candidate and retain adapter-plus-gate snapshots."""

    def __init__(
        self,
        *,
        candidate_bundle: ModelBundle,
        source_bundle: ModelBundle,
        gated_model: CausalPreservingGatedModel,
        validation_plan: Path,
        evaluation_batch_size: int,
        output_dir: Path,
        run: Any,
    ) -> None:
        self.candidate_bundle = candidate_bundle
        self.source_bundle = source_bundle
        self.gated_model = gated_model
        self.validation_plan = validation_plan
        self.evaluation_batch_size = evaluation_batch_size
        self.output_dir = output_dir
        self.run = run
        self.saved: set[int] = set()
        self.score_frames: list[pd.DataFrame] = []
        self.retained: list[dict[str, int | str | None]] = []
        self.source_scores: pd.DataFrame | None = None

    def _evaluate_and_retain(self, step: int) -> None:
        if step in self.saved:
            return
        assert_budget_reserve()
        with sdpa_kernel([SDPBackend.MATH]):
            scores = _evaluate_preserving_mode(
                self.candidate_bundle,
                validation_plan=self.validation_plan,
                batch_size=self.evaluation_batch_size,
                readout=f"gated_full_step{step:04d}",
                attention_mode="full",
            )
        if step == 0 and (
            self.source_scores is None or not _same_paired_scores(self.source_scores, scores)
        ):
            raise RuntimeError("zero-initialized gated candidate is not exactly causal")
        scores["optimizer_step"] = step
        self.score_frames.append(scores)
        summary = summarize_readouts(scores).iloc[0]
        gate_weight = self.gated_model.mixing_gate.weight.detach().float()
        self.run.log(
            {
                "gated_gate/step": step,
                "gated_gate/nucleotide_ce": float(summary["nucleotide_ce"]),
                "gated_gate/nucleotide_accuracy": float(summary["nucleotide_accuracy"]),
                "gated_gate/gate_weight_norm": float(gate_weight.norm()),
                "gated_gate/gate_weight_minimum": float(gate_weight.min()),
                "gated_gate/gate_weight_maximum": float(gate_weight.max()),
            }
        )

        snapshot_dir = self.output_dir / "gated-adapters" / f"step-{step:04d}"
        adapter_dir = snapshot_dir / "adapter"
        self.gated_model.peft_model.save_pretrained(adapter_dir, safe_serialization=True)
        save_file(
            {"mixing_gate.weight": gate_weight.cpu().contiguous()},
            snapshot_dir / "mixing_gate.safetensors",
        )
        artifact = wandb.Artifact(
            f"{GATED_MODEL_ARTIFACT_PREFIX}-step-{step:04d}",
            type="model",
            metadata={
                "optimizer_step": step,
                "format": "peft_adapter_plus_zero_initialized_logit_gate",
                "base_model": MODEL_ID,
                "base_revision": MODEL_REVISION,
                "mask_token": "[UNK]",
                "gate_weight_norm": float(gate_weight.norm()),
            },
        )
        artifact.add_dir(str(snapshot_dir), name="gated_adapter")
        logged = self.run.log_artifact(artifact, aliases=[f"step-{step:04d}"])
        logged.wait()
        self.retained.append(_artifact_record(logged, kind="peft_adapter_plus_gate", step=step))
        self.saved.add(step)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        with self.gated_model.peft_model.disable_adapter(), sdpa_kernel([SDPBackend.MATH]):
            self.source_scores = _evaluate_preserving_mode(
                self.source_bundle,
                validation_plan=self.validation_plan,
                batch_size=self.evaluation_batch_size,
                readout="source_causal_adapter_disabled_step0",
                attention_mode="causal",
            )
        self.source_scores["optimizer_step"] = 0
        self._evaluate_and_retain(0)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step in LONGRUN_CHECKPOINT_STEPS:
            self._evaluate_and_retain(step)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        self._evaluate_and_retain(int(trainer.global_step))


def _trajectory_tables(
    callback: RetainedGatedTrajectoryCallback,
    *,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if callback.source_scores is None:
        raise RuntimeError("gated trajectory lacks the source causal baseline")
    if callback.saved != set(LONGRUN_CHECKPOINT_STEPS):
        raise RuntimeError(f"gated trajectory checkpoints differ: {sorted(callback.saved)}")
    scores = pd.concat([callback.source_scores, *callback.score_frames], ignore_index=True)
    readout_identity = scores.groupby(["readout", "sample_id"]).size()
    target_identity = scores.groupby(["sample_id", "target_nucleotide_index"]).size()
    expected_readouts = len(LONGRUN_CHECKPOINT_STEPS) + 1
    if not (readout_identity == 1).all():
        raise RuntimeError("gated paired trajectory repeats a readout/sample target")
    if len(target_identity) != 640 or not (target_identity == expected_readouts).all():
        raise RuntimeError("gated readouts did not evaluate identical sample/target pairs")
    baseline = "source_causal_adapter_disabled_step0"
    summaries = [summarize_readouts(callback.source_scores).assign(optimizer_step=0)]
    comparisons: list[dict[str, object]] = []
    for step in LONGRUN_CHECKPOINT_STEPS:
        candidate = f"gated_full_step{step:04d}"
        candidate_scores = scores[scores["readout"] == candidate]
        summaries.append(summarize_readouts(candidate_scores).assign(optimizer_step=step))
        comparison = paired_comparison(
            scores[scores["readout"].isin((baseline, candidate))],
            candidate=candidate,
            baseline=baseline,
            n_bootstrap=n_bootstrap,
        )
        comparison["optimizer_step"] = step
        comparisons.append(comparison)
    summary = pd.concat(summaries, ignore_index=True)
    comparison_frame = pd.DataFrame(comparisons)
    return scores, summary, comparison_frame, information_gate(comparisons[-1])


def plot_gated_trajectory(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot exact paired validation trajectories against the frozen causal source."""

    source = summary[summary["readout"] == "source_causal_adapter_disabled_step0"].iloc[0]
    candidate = summary[summary["readout"].str.startswith("gated_full_step")].sort_values(
        "optimizer_step"
    )
    ordered = comparisons.sort_values("optimizer_step")
    steps = ordered["optimizer_step"].to_numpy(dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    axes[0, 0].plot(
        candidate["optimizer_step"], candidate["nucleotide_ce"], marker="o", color="#59A14F"
    )
    axes[0, 0].axhline(float(source["nucleotide_ce"]), color="#4C78A8", linestyle="--")
    axes[0, 0].set_ylabel("Four-way nucleotide CE")
    axes[0, 0].set_title("Gated full path vs frozen causal source")
    axes[0, 1].plot(
        candidate["optimizer_step"],
        candidate["nucleotide_accuracy"],
        marker="o",
        color="#59A14F",
    )
    axes[0, 1].axhline(float(source["nucleotide_accuracy"]), color="#4C78A8", linestyle="--")
    axes[0, 1].set_ylabel("Four-way nucleotide accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Same 640 single-mask targets")

    ce = ordered["nucleotide_ce_delta"].to_numpy(dtype=float)
    axes[1, 0].plot(steps, ce, marker="o", color="#59A14F")
    axes[1, 0].fill_between(
        steps,
        ordered["nucleotide_ce_delta_ci95_low"].to_numpy(dtype=float),
        ordered["nucleotide_ce_delta_ci95_high"].to_numpy(dtype=float),
        color="#59A14F",
        alpha=0.2,
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_ylabel("Paired CE delta vs source")
    accuracy = ordered["nucleotide_accuracy_delta"].to_numpy(dtype=float)
    axes[1, 1].plot(steps, accuracy, marker="o", color="#59A14F")
    axes[1, 1].fill_between(
        steps,
        ordered["nucleotide_accuracy_delta_ci95_low"].to_numpy(dtype=float),
        ordered["nucleotide_accuracy_delta_ci95_high"].to_numpy(dtype=float),
        color="#59A14F",
        alpha=0.2,
    )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_ylabel("Paired accuracy delta vs source")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
        axis.axvline(100, color="#999999", linestyle=":", linewidth=1)
        axis.axvline(800, color="#999999", linestyle=":", linewidth=1)
    figure.suptitle("Causal-preserving gated full-attention LoRA: validation trajectory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_gated_training(trace: pd.DataFrame, output_path: Path) -> None:
    """Plot gated/branch loss, gate opening, and separated gradient norms."""

    figure, axes = plt.subplots(3, 1, figsize=(7.2, 10), sharex=True, constrained_layout=True)
    for column, label, color in (
        ("gated_loss", "Gated candidate", "#4C78A8"),
        ("branch_loss", "Full-attention LoRA branch", "#E45756"),
    ):
        axes[0].plot(
            trace["step"],
            trace[column].rolling(50, min_periods=1).mean(),
            label=label,
            color=color,
        )
    axes[0].set_ylabel("50-step mean CE")
    axes[0].set_title("Training objectives")
    axes[0].legend()
    axes[1].plot(trace["step"], trace["gate_mean"], label="Mean signed gate", color="#59A14F")
    axes[1].plot(
        trace["step"],
        trace["gate_mean_absolute"],
        label="Mean absolute gate",
        color="#F28E2B",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Selected-target coefficient")
    axes[1].set_title("Right-context branch opening")
    axes[1].legend()
    axes[2].plot(trace["step"], trace["lora_gradient_norm"], label="LoRA", color="#B279A2")
    axes[2].plot(trace["step"], trace["gate_gradient_norm"], label="Gate", color="#76B7B2")
    axes[2].set_yscale("symlog", linthresh=1e-8)
    axes[2].set_ylabel("Pre-clip gradient L2 norm")
    axes[2].set_xlabel("Optimizer step")
    axes[2].set_title("Trainable-path gradients")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Causal-preserving dual-path training diagnostics")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_gated_lora_mntp(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    seed: int,
    num_workers: int,
    evaluation_batch_size: int,
    n_bootstrap: int,
) -> None:
    """Train and gate a separate full-attention LoRA path against the frozen source."""

    numeric_controls = configure_gated_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("gated LoRA MNTP experiment requires one CUDA GPU")
    if evaluation_batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("evaluation batch size and bootstrap count must be positive")
    assert_plan_contract(train_plan, validation_plan)
    if plan_sha256(validation_plan) != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("gated LoRA validation plan differs from the paired information gate")
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.006"))
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    if prior_cost + GATED_MAX_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("gated LoRA projection reaches the issue budget cap")

    output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = output_dir / "prelaunch-budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "prior_cost_usd": prior_cost,
                "maximum_instance_hours": GATED_MAX_INSTANCE_HOURS,
                "price_per_hour_usd": price,
                "projected_total_usd": prior_cost + GATED_MAX_INSTANCE_HOURS * price,
                "budget_cap_usd": BUDGET_USD,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    L.seed_everything(seed, workers=True)
    config = GatedLoraConfig()
    lora_config = LoraMntpConfig(
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        mask_probability=config.mask_probability,
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        cooldown_start_step=config.cooldown_start_step,
        train_steps=config.train_steps,
        microbatch_size=config.microbatch_size,
        accumulation_steps=config.accumulation_steps,
    )
    lora_bundle, _ = build_lora_bundle(lora_config)
    if not isinstance(lora_bundle.model, PeftModel):
        raise TypeError("LoRA builder did not return a PEFT model")
    lora_bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora_bundle.model.enable_input_require_grads()
    lora_bundle.model.config.use_cache = False
    gated_model = CausalPreservingGatedModel(
        lora_bundle.model,
        vocab_size=len(lora_bundle.tokenizer),
    )
    trainable_counts = assert_gated_trainables(gated_model)
    candidate_bundle = ModelBundle(
        model=gated_model,  # type: ignore[arg-type]
        tokenizer=lora_bundle.tokenizer,
        canonical_token_ids=lora_bundle.canonical_token_ids,
        mask_token_id=lora_bundle.mask_token_id,
        input_output_tied=lora_bundle.input_output_tied,
    )
    source_bundle = ModelBundle(
        model=FloatLogitView(lora_bundle.model),  # type: ignore[arg-type]
        tokenizer=lora_bundle.tokenizer,
        canonical_token_ids=lora_bundle.canonical_token_ids,
        mask_token_id=lora_bundle.mask_token_id,
        input_output_tied=lora_bundle.input_output_tied,
    )
    module = GatedLoraModule(model=gated_model, config=config)
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=lora_bundle.tokenizer,
        objective="mntp",
        canonical_token_ids=lora_bundle.canonical_token_ids,
        mask_token_id=lora_bundle.mask_token_id,
        batch_size=config.microbatch_size,
        seed=seed,
        num_workers=num_workers,
        fixed_mask_probability=config.mask_probability,
    )
    logger = WandbLogger(
        project=WANDB_PROJECT,
        group=GATED_WANDB_GROUP,
        name=GATED_RUN_NAME,
        tags=[
            *EXPERIMENT_TAGS,
            "causal-preserving",
            "dual-path",
            "zero-initialized-gate",
            "lora",
            "rank-16",
            "unk-mask",
            "paired-information-gate",
        ],
        save_dir=str(output_dir),
        log_model=False,
    )
    logger.log_hyperparams(
        config.to_dict()
        | {
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "mask_token": "[UNK]",
            "mask_replacement": "100% selected targets replaced",
            "effective_batch_size": LORA_EFFECTIVE_BATCH_SIZE,
            "trainable_parameters": trainable_counts,
            "numeric_controls": numeric_controls,
            "references": [DECTOENC_REFERENCE, BITUNE_REFERENCE],
        }
    )
    run = logger.experiment
    trajectory = RetainedGatedTrajectoryCallback(
        candidate_bundle=candidate_bundle,
        source_bundle=source_bundle,
        gated_model=gated_model,
        validation_plan=validation_plan,
        evaluation_batch_size=evaluation_batch_size,
        output_dir=output_dir,
        run=run,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=config.train_steps,
        max_epochs=-1,
        accumulate_grad_batches=config.accumulation_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=True,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=[
            trajectory,
            RuntimeMetricsCallback(output_dir / "runtime.json", LORA_EFFECTIVE_BATCH_SIZE),
            BudgetGuardCallback(
                instance_start_unix=(
                    None
                    if os.getenv("EXP479_INSTANCE_START_UNIX") is None
                    else float(os.environ["EXP479_INSTANCE_START_UNIX"])
                ),
                prior_cost_usd=prior_cost,
                price_per_hour_usd=price,
            ),
        ],
        enable_checkpointing=False,
    )

    try:
        trainer.fit(module, datamodule=data)
        if trainer.global_step != config.train_steps:
            raise RuntimeError(
                f"gated LoRA stopped at step {trainer.global_step}, expected {config.train_steps}"
            )
        scores, summary, comparisons, source_gate = _trajectory_tables(
            trajectory,
            n_bootstrap=n_bootstrap,
        )
        with sdpa_kernel([SDPBackend.MATH]):
            causalized = _evaluate_preserving_mode(
                candidate_bundle,
                validation_plan=validation_plan,
                batch_size=evaluation_batch_size,
                readout="gated_causalized_step1000",
                attention_mode="causal",
            )
        final_full = scores[scores["readout"] == "gated_full_step1000"].copy()
        right_context_comparison = paired_comparison(
            pd.concat([final_full, causalized], ignore_index=True),
            candidate="gated_full_step1000",
            baseline="gated_causalized_step1000",
            n_bootstrap=n_bootstrap,
        )
        right_context_gate = right_context_use_gate(right_context_comparison)
        overall_gate = {
            "criterion": (
                "candidate passes source non-inferiority and full attention is strictly useful "
                "relative to the same trained candidate with right context removed"
            ),
            "source_noninferiority": source_gate,
            "right_context_use": right_context_gate,
            "passed": bool(source_gate["passed"] and right_context_gate["passed"]),
        }

        with gated_model.peft_model.disable_adapter(), sdpa_kernel([SDPBackend.MATH]):
            final_source = _evaluate_preserving_mode(
                source_bundle,
                validation_plan=validation_plan,
                batch_size=evaluation_batch_size,
                readout="source_causal_adapter_disabled_step1000",
                attention_mode="causal",
            )
        if trajectory.source_scores is None:
            raise RuntimeError("gated source preservation check lacks step-0 scores")
        source_preserved = _same_paired_scores(trajectory.source_scores, final_source)
        if not source_preserved:
            raise RuntimeError("frozen causal source changed during gated LoRA training")

        all_scores = pd.concat([scores, causalized], ignore_index=True)
        scores_path = output_dir / "paired-nucleotide-scores.csv"
        summary_path = output_dir / "paired-nucleotide-summary.csv"
        comparisons_path = output_dir / "paired-nucleotide-comparisons.csv"
        right_context_path = output_dir / "right-context-comparison.json"
        gate_path = output_dir / "paired-nucleotide-gate.json"
        all_scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        comparisons.to_csv(comparisons_path, index=False)
        right_context_path.write_text(
            json.dumps(right_context_comparison, indent=2) + "\n",
            encoding="utf-8",
        )
        gate_path.write_text(json.dumps(overall_gate, indent=2) + "\n", encoding="utf-8")
        trajectory_figure = output_dir / "figures" / "paired-nucleotide-trajectory"
        plot_gated_trajectory(summary, comparisons, trajectory_figure)

        trace = pd.DataFrame(module.gradient_norm_trace)
        if trace["step"].astype(int).tolist() != list(range(config.train_steps)):
            raise RuntimeError("gated gradient trace omits an optimizer step")
        trace_path = output_dir / "gradient-norm-trace.csv"
        trace.to_csv(trace_path, index=False)
        stability_figure = output_dir / "figures" / "training-stability"
        plot_longrun_stability(
            trace,
            stability_figure,
            title="Causal-preserving gated LoRA training stability",
        )
        gate_figure = output_dir / "figures" / "gate-training"
        plot_gated_training(trace, gate_figure)

        checkpoint_path = output_dir / "checkpoints" / "step-1000-gated-adapter-optimizer.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        torch.save(
            {
                "global_step": int(trainer.global_step),
                "adapter_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in get_peft_model_state_dict(gated_model.peft_model).items()
                },
                "mixing_gate_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in gated_model.mixing_gate.state_dict().items()
                },
                "optimizer_state_dict": trainer.optimizers[0].state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "train_plan_sha256": plan_sha256(train_plan),
                "validation_plan_sha256": plan_sha256(validation_plan),
                "next_training_sample_id": config.train_steps * LORA_EFFECTIVE_BATCH_SIZE,
                "config": config.to_dict(),
            },
            checkpoint_path,
        )
        checkpoint_artifact = wandb.Artifact(
            f"{GATED_MODEL_ARTIFACT_PREFIX}-step-1000-optimizer",
            type="model",
            metadata={
                "optimizer_step": 1_000,
                "format": "adapter_gate_optimizer_rng",
                "contains_optimizer_state": True,
                "base_weights_included": False,
            },
        )
        checkpoint_artifact.add_file(str(checkpoint_path))
        logged_checkpoint = run.log_artifact(
            checkpoint_artifact,
            aliases=["step-1000-optimizer"],
        )
        logged_checkpoint.wait()
        trajectory.retained.append(
            _artifact_record(
                logged_checkpoint,
                kind="adapter_gate_optimizer_checkpoint",
                step=1_000,
            )
        )
        retention_path = output_dir / "retention-manifest.json"
        _write_retention_manifest(retention_path, trajectory.retained)

        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest = {
            "status": "completed",
            "run_name": GATED_RUN_NAME,
            "wandb_url": run.get_url(),
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "base_frozen": True,
            "disabled_adapter_causal_bit_exact": source_preserved,
            "step0_candidate_causal_bit_exact": True,
            "trainable_parameters": trainable_counts,
            "configuration": config.to_dict(),
            "numeric_controls": numeric_controls,
            "effective_batch_size": LORA_EFFECTIVE_BATCH_SIZE,
            "mask_token": "[UNK]",
            "mask_replacement": "100% selected targets replaced",
            "training_objective": (
                "gated candidate sequence-balanced repeat-weight MNTP CE plus an equally "
                "weighted auxiliary full-branch MNTP CE; both retain source z-loss"
            ),
            "architecture": (
                "frozen adapter-disabled causal logits plus tanh of a zero-initialized "
                "source-logit projection times the separate full-attention LoRA logit delta"
            ),
            "references": [DECTOENC_REFERENCE, BITUNE_REFERENCE],
            "train_plan_sha256": plan_sha256(train_plan),
            "validation_plan_sha256": plan_sha256(validation_plan),
            "paired_target_count": 640,
            "gate": overall_gate,
            "right_context_comparison": right_context_comparison,
            "checkpoint_retention": "adapter-plus-gate and optimizer-bearing W&B artifacts",
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
            "vep_evaluation": "not performed",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run.log(
            {
                "gated_gate/summary": wandb.Table(dataframe=summary),
                "gated_gate/comparisons": wandb.Table(dataframe=comparisons),
                "gated_gate/trajectory": wandb.Image(str(trajectory_figure.with_suffix(".png"))),
                "gated_gate/stability": wandb.Image(str(stability_figure.with_suffix(".png"))),
                "gated_gate/training": wandb.Image(str(gate_figure.with_suffix(".png"))),
            }
        )
        run.summary["gated_gate/passed"] = bool(overall_gate["passed"])
        run.summary["gated_gate/source_noninferiority_passed"] = bool(source_gate["passed"])
        run.summary["gated_gate/right_context_use_passed"] = bool(right_context_gate["passed"])
        run.summary["gated_gate/source_causal_preserved"] = source_preserved
        result_artifact = wandb.Artifact(GATED_EVALUATION_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            right_context_path,
            gate_path,
            trace_path,
            output_dir / "runtime.json",
            budget_path,
            retention_path,
            manifest_path,
            cost_path,
            trajectory_figure.with_suffix(".svg"),
            stability_figure.with_suffix(".svg"),
            gate_figure.with_suffix(".svg"),
        ):
            result_artifact.add_file(str(path))
        logged_result = run.log_artifact(
            result_artifact,
            aliases=["paired-gate", "step-1000"],
        )
        logged_result.wait()
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        del trainer, data, module, candidate_bundle, source_bundle, gated_model, lora_bundle
        gc.collect()
        torch.cuda.empty_cache()
