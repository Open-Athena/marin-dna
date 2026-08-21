"""One-thousand-step AdamW causal trajectory for issue 479."""

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
from torch import nn
from transformers import PreTrainedModel

from exp479_mntp.callbacks import BudgetGuardCallback, RuntimeMetricsCallback
from exp479_mntp.causal_calibration import evaluate_causal_validation
from exp479_mntp.checkpoint_audit import (
    StepExportCallback,
    _loaded_from_hf,
    assert_plan_contract,
)
from exp479_mntp.config import (
    BUDGET_USD,
    EXPERIMENT_TAGS,
    LAMBDA_GH200_PRICE_PER_HOUR_USD,
    SOURCE_Z_LOSS_WEIGHT,
    WANDB_PROJECT,
    wsd_multiplier,
)
from exp479_mntp.data import plan_sha256
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.modeling import load_model_bundle
from exp479_mntp.module import AdaptationModule
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

LONGRUN_STEPS = 1_000
LONGRUN_LEARNING_RATE = 1e-5
LONGRUN_WARMUP_STEPS = 100
LONGRUN_COOLDOWN_START_STEP = 800
LONGRUN_CHECKPOINT_STEPS = (
    0,
    25,
    50,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1_000,
)
LONGRUN_MAX_GRAD_NORM = 1.0
LONGRUN_MAX_INSTANCE_HOURS = 2.0
LONGRUN_WANDB_GROUP = "dna-exp479-causal-longrun-corrected"
LONGRUN_RUN_NAME = "dna-exp479-clm-adamw-1e-5-corrected-wsd1000-seed0"
LONGRUN_MODEL_ARTIFACT_PREFIX = "dna-exp479-causal-longrun-corrected"
LONGRUN_EVALUATION_ARTIFACT = "dna-exp479-causal-longrun-corrected-lr1e-5"


@dataclass(frozen=True)
class AdamWLongRunConfig:
    """Optimizer and WSD schedule selected for the 1,000-step causal run."""

    learning_rate: float = LONGRUN_LEARNING_RATE
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = LONGRUN_MAX_GRAD_NORM
    warmup_steps: int = LONGRUN_WARMUP_STEPS
    cooldown_start_step: int = LONGRUN_COOLDOWN_START_STEP
    train_steps: int = LONGRUN_STEPS

    def __post_init__(self) -> None:
        wsd_multiplier(
            0,
            warmup_steps=self.warmup_steps,
            cooldown_start_step=self.cooldown_start_step,
            total_steps=self.train_steps,
        )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable values."""

        return asdict(self)


class CausalLongRunModule(AdaptationModule):
    """Train causal CLM with one AdamW group and the selected WSD schedule."""

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        batch_size: int,
        config: AdamWLongRunConfig | None = None,
    ) -> None:
        config = AdamWLongRunConfig() if config is None else config
        super().__init__(
            model=model,
            arm="clm_continuation",
            batch_size=batch_size,
            train_steps=config.train_steps,
            record_gradient_norms=True,
        )
        self.longrun_config = config

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.longrun_config.learning_rate,
            betas=(self.longrun_config.beta1, self.longrun_config.beta2),
            eps=self.longrun_config.epsilon,
            weight_decay=self.longrun_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: wsd_multiplier(
                step,
                warmup_steps=self.longrun_config.warmup_steps,
                cooldown_start_step=self.longrun_config.cooldown_start_step,
                total_steps=self.longrun_config.train_steps,
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
        total_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.longrun_config.max_grad_norm,
            error_if_nonfinite=True,
        )
        if self._latest_train_loss is None:
            raise RuntimeError("gradient trace lacks the corresponding training loss")
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        if len(learning_rates) != 1:
            raise RuntimeError(f"expected one AdamW parameter group, found {len(learning_rates)}")
        norm = float(total_norm)
        self.gradient_norm_trace.append(
            {
                "step": int(self.global_step),
                "train_loss": self._latest_train_loss,
                "pre_clip_gradient_norm": norm,
                "clipped": int(norm > self.longrun_config.max_grad_norm),
                "learning_rate": learning_rates[0],
            }
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["exp479"]["causal_longrun"] = self.longrun_config.to_dict()


class RetainedStepExportCallback(StepExportCallback):
    """Save each trajectory export and retain it immediately as a W&B artifact."""

    def __init__(
        self,
        output_dir: Path,
        tokenizer: Any,
        steps: tuple[int, ...],
        run: Any,
    ) -> None:
        super().__init__(output_dir, tokenizer, steps)
        self.run = run
        self.retained: list[dict[str, int | str | None]] = []

    def _save(self, trainer: L.Trainer, pl_module: AdaptationModule) -> None:
        before = set(self.saved)
        super()._save(trainer, pl_module)
        for step in sorted(self.saved - before):
            assert_budget_reserve()
            artifact = wandb.Artifact(
                f"{LONGRUN_MODEL_ARTIFACT_PREFIX}-step-{step:04d}",
                type="model",
                metadata={
                    "optimizer_step": step,
                    "objective": "clm_corrected_repeat_weight",
                    "format": "hf",
                    "z_loss_weight": SOURCE_Z_LOSS_WEIGHT,
                },
            )
            artifact.add_dir(str(self.output_dir / f"step-{step:04d}"), name="hf")
            logged = self.run.log_artifact(artifact, aliases=[f"step-{step:04d}"])
            logged.wait()
            self.retained.append(_artifact_record(logged, kind="hf_export", step=step))


def _artifact_record(artifact: Any, *, kind: str, step: int) -> dict[str, int | str | None]:
    """Return stable W&B identifiers for a completed artifact upload."""

    return {
        "kind": kind,
        "step": step,
        "artifact_id": getattr(artifact, "id", None),
        "artifact_name": getattr(artifact, "name", None),
        "artifact_version": getattr(artifact, "version", None),
        "qualified_name": getattr(artifact, "qualified_name", None),
    }


def _write_retention_manifest(
    output_path: Path,
    records: list[dict[str, int | str | None]],
) -> None:
    """Record every completed W&B checkpoint artifact upload."""

    output_path.write_text(
        json.dumps(
            {
                "backend": "wandb",
                "deletion_performed": False,
                "artifacts": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def projected_longrun_cost(prior_cost_usd: float) -> float:
    """Return a conservative two-hour prelaunch projection."""

    return prior_cost_usd + LONGRUN_MAX_INSTANCE_HOURS * LAMBDA_GH200_PRICE_PER_HOUR_USD


def summarize_macro_trajectory(losses: pd.DataFrame) -> dict[str, Any]:
    """Summarize the five-component validation macro without subset gates."""

    required = {"step", "component", "loss", "n_rows"}
    if not required.issubset(losses.columns):
        raise ValueError(f"validation table lacks {sorted(required - set(losses.columns))}")
    if set(losses["component"]) != {"macro"}:
        raise ValueError("long-run validation table must contain macro rows only")
    ordered = losses.sort_values("step")
    if ordered["step"].astype(int).tolist() != list(LONGRUN_CHECKPOINT_STEPS):
        raise RuntimeError("macro validation trajectory omits a checkpoint")
    steps = ordered["step"].to_numpy(dtype=float)
    values = ordered["loss"].to_numpy(dtype=float)
    delta = float(values[-1] - values[0])
    slope = float(np.polyfit(steps, values, 1)[0])
    return {
        "passed": delta <= 0.0,
        "criterion": "step-1000 five-component macro CE <= step-0 macro CE",
        "step_0_loss": float(values[0]),
        "step_1000_loss": float(values[-1]),
        "delta": delta,
        "linear_slope_per_step": slope,
        "n_rows": int(ordered["n_rows"].iloc[0]),
    }


def plot_macro_validation(losses: pd.DataFrame, output_path: Path) -> None:
    """Render the equal-component causal validation CE trajectory."""

    ordered = losses.sort_values("step")
    figure, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    axis.plot(
        ordered["step"],
        ordered["loss"],
        marker="o",
        color="#1F4E79",
        linewidth=1.8,
        label="Five-component macro",
    )
    axis.axhline(
        float(ordered.iloc[0]["loss"]),
        color="0.4",
        linestyle="--",
        linewidth=1,
        label="Step 0",
    )
    axis.axvline(100, color="#999999", linestyle=":", linewidth=1)
    axis.axvline(800, color="#999999", linestyle=":", linewidth=1)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Validation cross-entropy")
    axis.set_title("Corrected fixed-plan validation macro")
    axis.grid(alpha=0.25)
    axis.legend(title="Trajectory")
    axis.set_box_aspect(1)
    figure.suptitle("AdamW 1e-5 causal fine-tuning with corrected loss")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_longrun_stability(trace: pd.DataFrame, output_path: Path) -> None:
    """Render dense training loss, learning rate, and pre-clipping gradient norm."""

    figure, axes = plt.subplots(3, 1, figsize=(6.4, 10), sharex=True, constrained_layout=True)
    axes[0].plot(
        trace["step"],
        trace["train_loss"],
        color="#4C78A8",
        alpha=0.35,
        linewidth=0.7,
        label="Per-step loss",
    )
    axes[0].plot(
        trace["step"],
        trace["train_loss"].rolling(50, min_periods=1).mean(),
        color="#1F4E79",
        linewidth=1.6,
        label="50-step mean",
    )
    axes[0].set_ylabel("Training cross-entropy")
    axes[0].set_title("Training loss")
    axes[0].legend(title="Trace")

    axes[1].plot(trace["step"], trace["learning_rate"], color="#54A24B", linewidth=1.4)
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Warmup-stable-decay schedule")

    axes[2].plot(
        trace["step"],
        trace["pre_clip_gradient_norm"],
        color="#F58518",
        linewidth=0.9,
        label="Pre-clip norm",
    )
    axes[2].axhline(
        LONGRUN_MAX_GRAD_NORM,
        color="#B22222",
        linestyle="--",
        linewidth=1,
        label="Clip threshold",
    )
    axes[2].set_xlabel("Optimizer step")
    axes[2].set_ylabel("Gradient L2 norm")
    axes[2].set_title("Gradient stability")
    axes[2].legend(title="Trace")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_box_aspect(1)
    figure.suptitle("Corrected-loss AdamW 1e-5 training stability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_causal_longrun(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    batch_size: int,
    seed: int,
    num_workers: int,
    offline_wandb: bool,
) -> None:
    """Train, retain, and evaluate the selected 1,000-step causal trajectory."""

    if not torch.cuda.is_available():
        raise RuntimeError("causal long run requires the Lambda GH200")
    if batch_size != 64:
        raise ValueError(f"causal long run must reuse audited batch size 64, got {batch_size}")
    assert_plan_contract(train_plan, validation_plan)
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    projection = projected_longrun_cost(prior_cost)
    if projection >= BUDGET_USD:
        raise RuntimeError(f"causal long-run projection ${projection:.2f} reaches the budget cap")
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = output_dir / "prelaunch-budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "prior_cost_usd": prior_cost,
                "maximum_instance_hours": LONGRUN_MAX_INSTANCE_HOURS,
                "price_per_hour_usd": LAMBDA_GH200_PRICE_PER_HOUR_USD,
                "projected_total_usd": projection,
                "budget_cap_usd": BUDGET_USD,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    config = AdamWLongRunConfig()
    bundle = load_model_bundle(
        initialization="transferred",
        add_mask=False,
        attention_implementation="sdpa",
    )
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.config.use_cache = False
    module = CausalLongRunModule(model=bundle.model, batch_size=batch_size, config=config)
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective="clm",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=None,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )
    logger = WandbLogger(
        project=WANDB_PROJECT,
        group=LONGRUN_WANDB_GROUP,
        name=LONGRUN_RUN_NAME,
        tags=[
            *EXPERIMENT_TAGS,
            "causal-longrun",
            "corrected-loss",
            "adamw",
            "lr-1e-5",
            "wandb-retained",
        ],
        save_dir=str(output_dir),
        offline=offline_wandb,
        log_model=False,
    )
    logger.log_hyperparams(config.to_dict())
    run = logger.experiment
    export = RetainedStepExportCallback(
        output_dir / "exports",
        bundle.tokenizer,
        LONGRUN_CHECKPOINT_STEPS[1:],
        run,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=LONGRUN_STEPS,
        max_epochs=-1,
        accumulate_grad_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=True,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=[
            export,
            RuntimeMetricsCallback(output_dir / "runtime.json", batch_size),
            BudgetGuardCallback(
                instance_start_unix=(
                    None
                    if os.getenv("EXP479_INSTANCE_START_UNIX") is None
                    else float(os.environ["EXP479_INSTANCE_START_UNIX"])
                ),
                prior_cost_usd=prior_cost,
            ),
        ],
        enable_checkpointing=False,
    )
    try:
        trainer.fit(module, datamodule=data)
        if trainer.global_step != LONGRUN_STEPS:
            raise RuntimeError(
                f"causal long run stopped at step {trainer.global_step}, expected {LONGRUN_STEPS}"
            )
        if export.saved != set(LONGRUN_CHECKPOINT_STEPS[1:]):
            raise RuntimeError(f"causal long-run exports differ: {sorted(export.saved)}")

        trace = pd.DataFrame(module.gradient_norm_trace)
        if trace["step"].astype(int).tolist() != list(range(LONGRUN_STEPS)):
            raise RuntimeError("causal long-run gradient trace omits an optimizer step")
        trace.to_csv(output_dir / "gradient-norm-trace.csv", index=False)
        plot_longrun_stability(trace, output_dir / "figures" / "training-stability")

        final_checkpoint = output_dir / "checkpoints" / "step-1000.ckpt"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(final_checkpoint)
        assert_budget_reserve()
        checkpoint_artifact = wandb.Artifact(
            f"{LONGRUN_MODEL_ARTIFACT_PREFIX}-step-1000-full",
            type="model",
            metadata={
                "optimizer_step": LONGRUN_STEPS,
                "objective": "clm_corrected_repeat_weight",
                "z_loss_weight": SOURCE_Z_LOSS_WEIGHT,
                "format": "lightning",
                "contains_optimizer_state": True,
            },
        )
        checkpoint_artifact.add_file(
            str(final_checkpoint),
            name="checkpoints/step-1000.ckpt",
        )
        logged_checkpoint = run.log_artifact(
            checkpoint_artifact,
            aliases=["step-1000-full"],
        )
        logged_checkpoint.wait()
        retention_records = [
            *export.retained,
            _artifact_record(logged_checkpoint, kind="lightning_checkpoint", step=1_000),
        ]
        retention_path = output_dir / "retention-manifest.json"
        _write_retention_manifest(retention_path, retention_records)

        del trainer, data, module, bundle
        gc.collect()
        torch.cuda.empty_cache()

        rows: list[dict[str, float | int | str]] = []
        for step in LONGRUN_CHECKPOINT_STEPS:
            assert_budget_reserve()
            if step == 0:
                point = load_model_bundle(
                    initialization="transferred",
                    add_mask=False,
                    attention_implementation="sdpa",
                    dtype=torch.bfloat16,
                )
                model = point.model
                tokenizer = point.tokenizer
                canonical_ids = point.canonical_token_ids
            else:
                point = _loaded_from_hf(output_dir / "exports" / f"step-{step:04d}", "clm")
                model = point.model
                tokenizer = point.tokenizer
                canonical_ids = point.canonical_ids
            model.to(device="cuda", dtype=torch.bfloat16).eval()
            evaluated = evaluate_causal_validation(
                step=step,
                model=model,
                tokenizer=tokenizer,
                canonical_ids=canonical_ids,
                validation_plan=validation_plan,
                batch_size=batch_size,
            )
            rows.extend(row for row in evaluated if row["component"] == "macro")
            del point, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        losses = pd.DataFrame(rows)
        losses.to_csv(output_dir / "validation-loss.csv", index=False)
        gate = summarize_macro_trajectory(losses)
        gate_path = output_dir / "gate-summary.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        plot_macro_validation(losses, output_dir / "figures" / "validation-trajectory")

        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        wandb_url = run.get_url()
        manifest = {
            "run_name": LONGRUN_RUN_NAME,
            "wandb_url": wandb_url,
            "optimizer": config.to_dict(),
            "batch_size": batch_size,
            "seed": seed,
            "checkpoint_steps": list(LONGRUN_CHECKPOINT_STEPS),
            "train_plan_sha256": plan_sha256(train_plan),
            "validation_plan_sha256": plan_sha256(validation_plan),
            "training_objective": "global effective-weight mean of CE plus source z-loss",
            "training_z_loss_weight": SOURCE_Z_LOSS_WEIGHT,
            "validation_objective": "pure CE with one repeat-weight application",
            "validation_scope": "equal macro of five 128-row component reducers",
            "model_artifact_prefix": LONGRUN_MODEL_ARTIFACT_PREFIX,
            "checkpoint_retention": "W&B model artifacts listed in retention-manifest.json",
            "checkpoint_deletion": "not performed",
            "gate": gate,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run.define_metric("causal_longrun_corrected/step")
        run.define_metric("causal_longrun_corrected/*", step_metric="causal_longrun_corrected/step")
        for row in losses.sort_values("step").itertuples(index=False):
            run.log(
                {
                    "causal_longrun_corrected/step": int(row.step),
                    "causal_longrun_corrected/validation/macro_ce": float(row.loss),
                }
            )
        run.log(
            {
                "causal_longrun_corrected/validation_table": wandb.Table(dataframe=losses),
                "causal_longrun_corrected/validation_figure": wandb.Image(
                    str(output_dir / "figures" / "validation-trajectory.png")
                ),
                "causal_longrun_corrected/stability_figure": wandb.Image(
                    str(output_dir / "figures" / "training-stability.png")
                ),
            }
        )
        run.summary["validation_gate_passed"] = bool(gate["passed"])
        run.summary["step_0_macro_ce"] = float(gate["step_0_loss"])
        run.summary["step_1000_macro_ce"] = float(gate["step_1000_loss"])
        run.summary["checkpoint_retention"] = "W&B model artifacts"
        artifact = wandb.Artifact(LONGRUN_EVALUATION_ARTIFACT, type="evaluation")
        for path in (
            output_dir / "validation-loss.csv",
            gate_path,
            output_dir / "gradient-norm-trace.csv",
            output_dir / "runtime.json",
            budget_path,
            retention_path,
            manifest_path,
            cost_path,
            output_dir / "figures" / "validation-trajectory.svg",
            output_dir / "figures" / "training-stability.svg",
        ):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish(exit_code=0)
    except BaseException:
        logger.experiment.finish(exit_code=1)
        raise
