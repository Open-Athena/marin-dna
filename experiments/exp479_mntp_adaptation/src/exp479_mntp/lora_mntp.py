"""Frozen-base rank-16 MNTP adaptation with an exact paired information gate."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from torch import nn

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
from exp479_mntp.masking import sample_seed
from exp479_mntp.modeling import ModelBundle, load_model_bundle, model_logits
from exp479_mntp.module import AdaptationModule
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
    information_gate,
    paired_comparison,
    summarize_readouts,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_MASK_PROBABILITY = 0.2
LORA_ATTENTION_ANNEAL_STEPS = 800
LORA_LEARNING_RATE = 1e-5
LORA_MICROBATCH_SIZE = 16
LORA_EFFECTIVE_BATCH_SIZE = 64
LORA_ACCUMULATION_STEPS = LORA_EFFECTIVE_BATCH_SIZE // LORA_MICROBATCH_SIZE
LORA_MAX_INSTANCE_HOURS = 4.0
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LORA_WANDB_GROUP = "dna-exp479-lora-mntp-information-gate"
LORA_RUN_NAME = "dna-exp479-lora-r16-mntp-unk-20pct-damagecal800-lr1e-5-wsd1000-seed0"
LORA_MODEL_ARTIFACT_PREFIX = "dna-exp479-lora-r16-mntp-unk"
LORA_EVALUATION_ARTIFACT = "dna-exp479-lora-r16-mntp-information-gate"
LORA_ATTENTION_CALIBRATION_TAG = "source-unk-zero-training-v1"
LORA_ATTENTION_CALIBRATION_PROBABILITIES = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
)
LORA_ATTENTION_CALIBRATION_CE_DEGRADATION_FRACTIONS = (
    0.0,
    0.32857232778195733,
    0.49099526592768916,
    0.6737672586505843,
    0.7759986276017637,
    0.8128203254429095,
    0.856183506930262,
    0.8869100893685018,
    0.9062062080291675,
    0.9330877228683793,
    0.9620857023479154,
    1.0,
)


@dataclass(frozen=True)
class LoraMntpConfig:
    """The single conservative LoRA pilot selected after the step-0 mask controls."""

    rank: int = LORA_RANK
    alpha: int = LORA_ALPHA
    dropout: float = LORA_DROPOUT
    mask_probability: float = LORA_MASK_PROBABILITY
    attention_anneal_steps: int = LORA_ATTENTION_ANNEAL_STEPS
    learning_rate: float = LORA_LEARNING_RATE
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    cooldown_start_step: int = 800
    train_steps: int = 1_000
    microbatch_size: int = LORA_MICROBATCH_SIZE
    accumulation_steps: int = LORA_ACCUMULATION_STEPS

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not 0 < self.mask_probability <= 1:
            raise ValueError("MNTP mask probability must be in (0, 1]")
        if not 0 < self.attention_anneal_steps <= self.train_steps:
            raise ValueError("attention annealing must finish within the training run")
        if self.learning_rate <= 0:
            raise ValueError("LoRA learning rate must be positive")
        if self.microbatch_size * self.accumulation_steps != LORA_EFFECTIVE_BATCH_SIZE:
            raise ValueError("LoRA microbatch and accumulation must preserve effective batch 64")
        wsd_multiplier(
            0,
            warmup_steps=self.warmup_steps,
            cooldown_start_step=self.cooldown_start_step,
            total_steps=self.train_steps,
        )

    def to_dict(self) -> dict[str, float | int | str | list[str] | list[float]]:
        """Return JSON-serializable configuration and target modules."""

        return asdict(self) | {
            "target_modules": list(LORA_TARGET_MODULES),
            "attention_schedule": "source_ce_damage_calibrated_piecewise_linear",
            "attention_calibration_tag": LORA_ATTENTION_CALIBRATION_TAG,
            "attention_calibration_probabilities": list(LORA_ATTENTION_CALIBRATION_PROBABILITIES),
            "attention_calibration_ce_degradation_fractions": list(
                LORA_ATTENTION_CALIBRATION_CE_DEGRADATION_FRACTIONS
            ),
        }


def damage_calibrated_future_edge_probability(target_ce_degradation_fraction: float) -> float:
    """Invert the measured frozen-source CE-damage curve by linear interpolation."""

    if not 0.0 <= target_ce_degradation_fraction <= 1.0:
        raise ValueError("target CE-degradation fraction must be in [0, 1]")
    return float(
        np.interp(
            target_ce_degradation_fraction,
            LORA_ATTENTION_CALIBRATION_CE_DEGRADATION_FRACTIONS,
            LORA_ATTENTION_CALIBRATION_PROBABILITIES,
        )
    )


def attention_future_edge_probability(step: int, *, anneal_steps: int) -> float:
    """Open future edges at an approximately linear frozen-source CE-damage rate."""

    if step < 0:
        raise ValueError("optimizer step must be non-negative")
    if anneal_steps <= 0:
        raise ValueError("attention anneal steps must be positive")
    if step >= anneal_steps:
        return 1.0
    return damage_calibrated_future_edge_probability(step / anneal_steps)


def annealed_attention_mask(
    attention_mask: torch.Tensor,
    *,
    future_edge_probability: float,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Open future attention edges with the released DiffuLLaMA recipe.

    The causal lower triangle is always open. Every other edge is sampled once
    and the same sampled matrix is shared across the batch, matching the paper's
    released implementation. The 2D token mask is still enforced on key positions.
    """

    if attention_mask.ndim != 2:
        raise ValueError("token attention mask must have shape [batch, sequence]")
    if not 0.0 <= future_edge_probability <= 1.0:
        raise ValueError("future-edge probability must be in [0, 1]")
    if not dtype.is_floating_point:
        raise ValueError("annealed additive attention mask requires a floating dtype")

    batch_size, sequence_length = attention_mask.shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    causal = torch.ones((sequence_length, sequence_length), dtype=torch.bool).tril()
    if future_edge_probability == 0.0:
        random_edges = torch.zeros_like(causal)
    elif future_edge_probability == 1.0:
        random_edges = torch.ones_like(causal)
    else:
        random_edges = (
            torch.rand((sequence_length, sequence_length), generator=generator)
            < future_edge_probability
        )
    allowed = torch.logical_or(causal, random_edges).to(attention_mask.device)
    allowed = allowed[None, None, :, :].expand(batch_size, 1, -1, -1)
    allowed = torch.logical_and(
        allowed,
        attention_mask.to(dtype=torch.bool)[:, None, None, :],
    )
    additive = torch.zeros(allowed.shape, dtype=dtype, device=attention_mask.device)
    return additive.masked_fill(~allowed, torch.finfo(dtype).min)


def assert_lora_trainables(model: nn.Module) -> tuple[int, tuple[str, ...]]:
    """Require that only the registered LoRA matrices can receive updates."""

    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("LoRA model has no trainable parameters")
    unexpected = [name for name in trainable if ".lora_A." not in name and ".lora_B." not in name]
    if unexpected:
        raise RuntimeError(f"non-LoRA parameters are trainable: {unexpected[:5]}")
    missing_targets = [
        target
        for target in LORA_TARGET_MODULES
        if not any(f"{target}.lora_" in name for name in trainable)
    ]
    if missing_targets:
        raise RuntimeError(f"registered LoRA target modules are missing: {missing_targets}")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return count, trainable


def build_lora_bundle(config: LoraMntpConfig) -> tuple[ModelBundle, int]:
    """Load the untouched source model and add zero-initialized rank-16 adapters."""

    source = load_model_bundle(
        initialization="transferred",
        add_mask=False,
        attention_implementation="sdpa",
    )
    unk_token_id = source.tokenizer.unk_token_id
    if unk_token_id is None or int(unk_token_id) < 0:
        raise RuntimeError("source tokenizer lacks the selected existing UNK mask token")
    lora = get_peft_model(
        source.model,
        LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            target_modules=list(LORA_TARGET_MODULES),
            lora_dropout=config.dropout,
            bias="none",
            task_type=None,
        ),
    )
    trainable_count, _ = assert_lora_trainables(lora)
    return (
        ModelBundle(
            model=lora,
            tokenizer=source.tokenizer,
            canonical_token_ids=source.canonical_token_ids,
            mask_token_id=int(unk_token_id),
            input_output_tied=source.input_output_tied,
        ),
        trainable_count,
    )


class LoraMntpModule(AdaptationModule):
    """Train LoRA matrices while annealing causal attention to full attention."""

    def __init__(self, *, model: PeftModel, config: LoraMntpConfig, seed: int = 0) -> None:
        super().__init__(
            model=model,
            arm="transferred_mntp",
            batch_size=config.microbatch_size,
            train_steps=config.train_steps,
            warmup_steps=config.warmup_steps,
            cooldown_start_step=config.cooldown_start_step,
            record_gradient_norms=True,
        )
        self.lora_config = config
        self.attention_seed = seed
        self._latest_attention_future_edge_probability: float | None = None

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        probability = attention_future_edge_probability(
            int(self.global_step),
            anneal_steps=self.lora_config.attention_anneal_steps,
        )
        first_sample_id = int(batch["sample_ids"][0])
        mask = annealed_attention_mask(
            batch["attention_mask"],
            future_edge_probability=probability,
            seed=sample_seed(self.attention_seed, first_sample_id, stream=3),
            dtype=next(self.model.parameters()).dtype,
        )
        self._latest_attention_future_edge_probability = probability
        return model_logits(
            self.model,
            input_ids=batch["input_ids"],
            attention_mask=mask,
            attention_mode="full",
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        objective_loss = super().training_step(batch, batch_idx)
        probability = self._latest_attention_future_edge_probability
        if probability is None:
            raise RuntimeError("attention annealing trace lacks its training forward pass")
        self.log(
            "train/attention_future_edge_probability",
            probability,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        return objective_loss

    def configure_optimizers(self) -> dict[str, Any]:
        assert_lora_trainables(self.model)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lora_config.learning_rate,
            betas=(self.lora_config.beta1, self.lora_config.beta2),
            eps=self.lora_config.epsilon,
            weight_decay=self.lora_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: wsd_multiplier(
                step,
                warmup_steps=self.lora_config.warmup_steps,
                cooldown_start_step=self.lora_config.cooldown_start_step,
                total_steps=self.lora_config.train_steps,
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
        total_norm = nn.utils.clip_grad_norm_(
            parameters,
            self.lora_config.max_grad_norm,
            error_if_nonfinite=True,
        )
        if self._latest_train_loss is None:
            raise RuntimeError("LoRA gradient trace lacks its training loss")
        if self._latest_attention_future_edge_probability is None:
            raise RuntimeError("LoRA gradient trace lacks its attention annealing probability")
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        if len(learning_rates) != 1:
            raise RuntimeError(f"expected one LoRA optimizer group, found {len(learning_rates)}")
        norm = float(total_norm)
        self.gradient_norm_trace.append(
            {
                "step": int(self.global_step),
                "train_loss": self._latest_train_loss,
                "pre_clip_gradient_norm": norm,
                "clipped": int(norm > self.lora_config.max_grad_norm),
                "learning_rate": learning_rates[0],
                "attention_future_edge_probability": (
                    self._latest_attention_future_edge_probability
                ),
            }
        )
        self.log(
            "train/pre_clip_gradient_norm",
            norm,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/gradient_was_clipped",
            float(norm > self.lora_config.max_grad_norm),
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/learning_rate",
            learning_rates[0],
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )


def _evaluate_preserving_mode(
    bundle: ModelBundle,
    *,
    validation_plan: Path,
    batch_size: int,
    readout: str,
    attention_mode: str,
    evaluation_device: str | torch.device = "cuda",
) -> pd.DataFrame:
    bundle.model.to(device=evaluation_device)
    was_training = bundle.model.training
    scores = evaluate_readout(
        bundle,
        validation_plan=validation_plan,
        batch_size=batch_size,
        readout=readout,
        attention_mode=attention_mode,
    )
    bundle.model.train(was_training)
    return scores


class RetainedLoraTrajectoryCallback(L.Callback):
    """Evaluate exact paired targets and retain adapter-only trajectory snapshots."""

    def __init__(
        self,
        *,
        bundle: ModelBundle,
        validation_plan: Path,
        evaluation_batch_size: int,
        output_dir: Path,
        run: Any,
    ) -> None:
        self.bundle = bundle
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
        scores = _evaluate_preserving_mode(
            self.bundle,
            validation_plan=self.validation_plan,
            batch_size=self.evaluation_batch_size,
            readout=f"lora_full_step{step:04d}",
            attention_mode="full",
        )
        scores["optimizer_step"] = step
        self.score_frames.append(scores)
        summary = summarize_readouts(scores).iloc[0]
        self.run.log(
            {
                "lora_gate/step": step,
                "lora_gate/full_nucleotide_ce": float(summary["nucleotide_ce"]),
                "lora_gate/full_nucleotide_accuracy": float(summary["nucleotide_accuracy"]),
            }
        )

        adapter_dir = self.output_dir / "adapters" / f"step-{step:04d}"
        self.bundle.model.save_pretrained(adapter_dir, safe_serialization=True)
        artifact = wandb.Artifact(
            f"{LORA_MODEL_ARTIFACT_PREFIX}-step-{step:04d}",
            type="model",
            metadata={
                "optimizer_step": step,
                "format": "peft_adapter",
                "base_model": MODEL_ID,
                "base_revision": MODEL_REVISION,
                "mask_token": "[UNK]",
            },
        )
        artifact.add_dir(str(adapter_dir), name="adapter")
        logged = self.run.log_artifact(artifact, aliases=[f"step-{step:04d}"])
        logged.wait()
        self.retained.append(_artifact_record(logged, kind="peft_adapter", step=step))
        self.saved.add(step)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        model = self.bundle.model
        if not isinstance(model, PeftModel):
            raise TypeError("LoRA trajectory callback requires a PEFT model")
        with model.disable_adapter():
            self.source_scores = _evaluate_preserving_mode(
                self.bundle,
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
    callback: RetainedLoraTrajectoryCallback,
    *,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if callback.source_scores is None:
        raise RuntimeError("LoRA trajectory lacks the source causal baseline")
    if callback.saved != set(LONGRUN_CHECKPOINT_STEPS):
        raise RuntimeError(f"LoRA trajectory checkpoints differ: {sorted(callback.saved)}")
    scores = pd.concat([callback.source_scores, *callback.score_frames], ignore_index=True)
    readout_identity = scores.groupby(["readout", "sample_id"]).size()
    target_identity = scores.groupby(["sample_id", "target_nucleotide_index"]).size()
    expected_readouts = len(LONGRUN_CHECKPOINT_STEPS) + 1
    if not (readout_identity == 1).all():
        raise RuntimeError("LoRA paired trajectory repeats a readout/sample target")
    if len(target_identity) != 640 or not (target_identity == expected_readouts).all():
        raise RuntimeError("LoRA readouts did not evaluate identical sample/target pairs")
    summaries: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []
    baseline = "source_causal_adapter_disabled_step0"
    for step in LONGRUN_CHECKPOINT_STEPS:
        candidate = f"lora_full_step{step:04d}"
        selected = scores[scores["readout"].isin((baseline, candidate))]
        summary = summarize_readouts(selected[selected["readout"] == candidate])
        summary["optimizer_step"] = step
        summaries.append(summary)
        comparison = paired_comparison(
            selected,
            candidate=candidate,
            baseline=baseline,
            n_bootstrap=n_bootstrap,
        )
        comparison["optimizer_step"] = step
        comparisons.append(comparison)
    source_summary = summarize_readouts(callback.source_scores)
    source_summary["optimizer_step"] = 0
    summary_frame = pd.concat([source_summary, *summaries], ignore_index=True)
    comparison_frame = pd.DataFrame(comparisons)
    gate = information_gate(comparisons[-1])
    return scores, summary_frame, comparison_frame, gate


def plot_lora_trajectory(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_path: Path,
) -> None:
    """Show absolute and paired-delta information trajectories with the source gate."""

    source = summary[summary["readout"] == "source_causal_adapter_disabled_step0"].iloc[0]
    lora = summary[summary["readout"].str.startswith("lora_full_step")].sort_values(
        "optimizer_step"
    )
    ordered = comparisons.sort_values("optimizer_step")
    steps = ordered["optimizer_step"].to_numpy(dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    axes[0, 0].plot(lora["optimizer_step"], lora["nucleotide_ce"], marker="o", color="#E45756")
    axes[0, 0].axhline(float(source["nucleotide_ce"]), color="#4C78A8", linestyle="--")
    axes[0, 0].set_ylabel("Four-way nucleotide CE")
    axes[0, 0].set_title("Full-attention LoRA vs frozen causal source")
    axes[0, 1].plot(
        lora["optimizer_step"],
        lora["nucleotide_accuracy"],
        marker="o",
        color="#E45756",
    )
    axes[0, 1].axhline(float(source["nucleotide_accuracy"]), color="#4C78A8", linestyle="--")
    axes[0, 1].set_ylabel("Four-way nucleotide accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Same 640 target nucleotides")

    ce_delta = ordered["nucleotide_ce_delta"].to_numpy(dtype=float)
    axes[1, 0].plot(steps, ce_delta, marker="o", color="#E45756")
    axes[1, 0].fill_between(
        steps,
        ordered["nucleotide_ce_delta_ci95_low"].to_numpy(dtype=float),
        ordered["nucleotide_ce_delta_ci95_high"].to_numpy(dtype=float),
        color="#E45756",
        alpha=0.2,
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_ylabel("Paired CE delta vs source")
    axes[1, 0].set_title("Gate passes at or below zero")
    accuracy_delta = ordered["nucleotide_accuracy_delta"].to_numpy(dtype=float)
    axes[1, 1].plot(steps, accuracy_delta, marker="o", color="#E45756")
    axes[1, 1].fill_between(
        steps,
        ordered["nucleotide_accuracy_delta_ci95_low"].to_numpy(dtype=float),
        ordered["nucleotide_accuracy_delta_ci95_high"].to_numpy(dtype=float),
        color="#E45756",
        alpha=0.2,
    )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_ylabel("Paired accuracy delta vs source")
    axes[1, 1].set_title("Gate passes at or above zero")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
        axis.axvline(100, color="#999999", linestyle=":", linewidth=1)
        axis.axvline(800, color="#999999", linestyle=":", linewidth=1)
    figure.suptitle("Frozen-base rank-16 LoRA: paired nucleotide information gate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_lora_mntp(
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
    """Train one sequential LoRA pilot and apply the exact paired information gate."""

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA MNTP pilot requires one CUDA GPU")
    if evaluation_batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("evaluation batch size and bootstrap count must be positive")
    assert_plan_contract(train_plan, validation_plan)
    if plan_sha256(validation_plan) != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("LoRA pilot validation plan differs from the paired information gate")
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.29"))
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    if prior_cost + LORA_MAX_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("LoRA pilot projection reaches the issue budget cap")

    output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = output_dir / "prelaunch-budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "prior_cost_usd": prior_cost,
                "maximum_instance_hours": LORA_MAX_INSTANCE_HOURS,
                "price_per_hour_usd": price,
                "projected_total_usd": prior_cost + LORA_MAX_INSTANCE_HOURS * price,
                "budget_cap_usd": BUDGET_USD,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    config = LoraMntpConfig()
    bundle, trainable_count = build_lora_bundle(config)
    if not isinstance(bundle.model, PeftModel):
        raise TypeError("LoRA builder did not return a PEFT model")
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.enable_input_require_grads()
    bundle.model.config.use_cache = False
    module = LoraMntpModule(model=bundle.model, config=config, seed=seed)
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective="mntp",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id,
        batch_size=config.microbatch_size,
        seed=seed,
        num_workers=num_workers,
        fixed_mask_probability=config.mask_probability,
    )
    logger = WandbLogger(
        project=WANDB_PROJECT,
        group=LORA_WANDB_GROUP,
        name=LORA_RUN_NAME,
        tags=[
            *EXPERIMENT_TAGS,
            "lora",
            "rank-16",
            "unk-mask",
            "damage-calibrated-attention",
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
            "trainable_parameters": trainable_count,
        }
    )
    run = logger.experiment
    trajectory = RetainedLoraTrajectoryCallback(
        bundle=bundle,
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
                f"LoRA pilot stopped at step {trainer.global_step}, expected {config.train_steps}"
            )
        scores, summary, comparisons, gate = _trajectory_tables(
            trajectory,
            n_bootstrap=n_bootstrap,
        )

        with bundle.model.disable_adapter():
            final_disabled = _evaluate_preserving_mode(
                bundle,
                validation_plan=validation_plan,
                batch_size=evaluation_batch_size,
                readout="source_causal_adapter_disabled_step1000",
                attention_mode="causal",
            )
        if trajectory.source_scores is None:
            raise RuntimeError("LoRA source preservation check lacks step-0 scores")
        preservation_columns = (
            "sample_id",
            "target_nucleotide_index",
            "nucleotide_ce",
            "nucleotide_correct",
            "full_vocab_ce",
            "full_vocab_correct",
        )
        source_preserved = all(
            np.array_equal(
                trajectory.source_scores[column].to_numpy(),
                final_disabled[column].to_numpy(),
            )
            for column in preservation_columns
        )
        if not source_preserved:
            raise RuntimeError("disabled-adapter causal readout changed despite a frozen base")

        scores_path = output_dir / "paired-nucleotide-scores.csv"
        summary_path = output_dir / "paired-nucleotide-summary.csv"
        comparisons_path = output_dir / "paired-nucleotide-comparisons.csv"
        gate_path = output_dir / "paired-nucleotide-gate.json"
        scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        comparisons.to_csv(comparisons_path, index=False)
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        trajectory_figure = output_dir / "figures" / "paired-nucleotide-trajectory"
        plot_lora_trajectory(summary, comparisons, trajectory_figure)

        trace = pd.DataFrame(module.gradient_norm_trace)
        if trace["step"].astype(int).tolist() != list(range(config.train_steps)):
            raise RuntimeError("LoRA gradient trace omits an optimizer step")
        trace_path = output_dir / "gradient-norm-trace.csv"
        trace.to_csv(trace_path, index=False)
        stability_figure = output_dir / "figures" / "training-stability"
        plot_longrun_stability(
            trace,
            stability_figure,
            title="Frozen-base rank-16 LoRA MNTP training stability",
        )

        checkpoint_path = output_dir / "checkpoints" / "step-1000-adapter-optimizer.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        torch.save(
            {
                "global_step": int(trainer.global_step),
                "adapter_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in get_peft_model_state_dict(bundle.model).items()
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
            f"{LORA_MODEL_ARTIFACT_PREFIX}-step-1000-optimizer",
            type="model",
            metadata={
                "optimizer_step": 1_000,
                "format": "adapter_optimizer_rng",
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
            _artifact_record(logged_checkpoint, kind="adapter_optimizer_checkpoint", step=1_000)
        )
        retention_path = output_dir / "retention-manifest.json"
        _write_retention_manifest(retention_path, trajectory.retained)

        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest = {
            "status": "completed",
            "run_name": LORA_RUN_NAME,
            "wandb_url": run.get_url(),
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "base_frozen": True,
            "disabled_adapter_causal_bit_exact": source_preserved,
            "trainable_parameters": trainable_count,
            "configuration": config.to_dict(),
            "effective_batch_size": LORA_EFFECTIVE_BATCH_SIZE,
            "mask_token": "[UNK]",
            "mask_replacement": "100% selected targets replaced",
            "training_objective": (
                "sequence-balanced effective-repeat-weight MNTP CE plus source z-loss"
            ),
            "attention_training": (
                "source-CE-damage-calibrated stochastic future-edge annealing over steps "
                "0-800, then full"
            ),
            "train_plan_sha256": plan_sha256(train_plan),
            "validation_plan_sha256": plan_sha256(validation_plan),
            "paired_target_count": 640,
            "gate": gate,
            "checkpoint_retention": "adapter-only and optimizer-bearing W&B artifacts",
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
                "lora_gate/summary": wandb.Table(dataframe=summary),
                "lora_gate/comparisons": wandb.Table(dataframe=comparisons),
                "lora_gate/trajectory": wandb.Image(str(trajectory_figure.with_suffix(".png"))),
                "lora_gate/stability": wandb.Image(str(stability_figure.with_suffix(".png"))),
            }
        )
        run.summary["lora_gate/passed"] = bool(gate["passed"])
        run.summary["lora_gate/point_estimate_passed"] = bool(gate["point_estimate_passed"])
        run.summary["lora_gate/source_causal_preserved"] = source_preserved
        result_artifact = wandb.Artifact(LORA_EVALUATION_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            gate_path,
            trace_path,
            output_dir / "runtime.json",
            budget_path,
            retention_path,
            manifest_path,
            cost_path,
            trajectory_figure.with_suffix(".svg"),
            stability_figure.with_suffix(".svg"),
        ):
            result_artifact.add_file(str(path))
        logged_result = run.log_artifact(result_artifact, aliases=["paired-gate", "step-1000"])
        logged_result.wait()
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        del trainer, data, module, bundle
        gc.collect()
        torch.cuda.empty_cache()
