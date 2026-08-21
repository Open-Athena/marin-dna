"""Conservative causal fine-tuning sanity gate for issue 479."""

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
from huggingface_hub import HfApi
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from exp479_mntp.callbacks import BudgetGuardCallback, RuntimeMetricsCallback
from exp479_mntp.checkpoint_audit import (
    CHECKPOINT_REPOS,
    StepExportCallback,
    _loaded_from_hf,
    assert_plan_contract,
)
from exp479_mntp.config import (
    BUDGET_USD,
    DATA_COMPONENTS,
    EXPERIMENT_TAGS,
    LAMBDA_GH200_PRICE_PER_HOUR_USD,
    WANDB_PROJECT,
)
from exp479_mntp.data import SequenceCollator, SequencePlanDataset, plan_sha256
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.loss import per_sequence_weighted_loss
from exp479_mntp.modeling import load_model_bundle, model_logits
from exp479_mntp.module import AdaptationModule
from exp479_mntp.publishing import assert_budget_reserve

CALIBRATION_STEPS = 200
CALIBRATION_LEARNING_RATE = 1e-6
CALIBRATION_WARMUP_STEPS = 10
CALIBRATION_CHECKPOINT_STEPS = (0, 1, 10, 25, 50, 100, 200)
CALIBRATION_MAX_GRAD_NORM = 1.0
CALIBRATION_MAX_INSTANCE_HOURS = 2.0
CALIBRATION_WANDB_GROUP = "dna-exp479-causal-calibration"
CALIBRATION_RUN_NAME = "dna-exp479-clm-adamw-1e-6-sanity-seed0"
CALIBRATION_REMOTE_PATH = "evaluation/causal-calibration-lr1e-6"
CALIBRATION_FINAL_MODEL_PATH = "hf/causal-calibration-lr1e-6/step-200"


@dataclass(frozen=True)
class AdamWCalibrationConfig:
    """Optimizer and schedule fixed for the first conservative sanity arm."""

    learning_rate: float = CALIBRATION_LEARNING_RATE
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = CALIBRATION_MAX_GRAD_NORM
    warmup_steps: int = CALIBRATION_WARMUP_STEPS
    train_steps: int = CALIBRATION_STEPS

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable values."""

        return asdict(self)


def warmup_constant_multiplier(step: int, *, warmup_steps: int) -> float:
    """Linearly warm from one fraction of peak LR, then remain constant."""

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
    return min(1.0, (step + 1) / warmup_steps)


class CausalCalibrationModule(AdaptationModule):
    """Train causal CLM with one ordinary AdamW parameter group."""

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        batch_size: int,
        config: AdamWCalibrationConfig | None = None,
    ) -> None:
        config = AdamWCalibrationConfig() if config is None else config
        super().__init__(
            model=model,
            arm="clm_continuation",
            batch_size=batch_size,
            train_steps=config.train_steps,
            record_gradient_norms=True,
        )
        self.calibration_config = config

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.calibration_config.learning_rate,
            betas=(self.calibration_config.beta1, self.calibration_config.beta2),
            eps=self.calibration_config.epsilon,
            weight_decay=self.calibration_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_constant_multiplier(
                step,
                warmup_steps=self.calibration_config.warmup_steps,
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
            self.calibration_config.max_grad_norm,
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
                "clipped": int(norm > self.calibration_config.max_grad_norm),
                "learning_rate": learning_rates[0],
            }
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["exp479"]["calibration"] = self.calibration_config.to_dict()


def _accumulate(
    totals: dict[str, dict[str, float]],
    key: str,
    *,
    loss: float,
    accuracy: float,
    count: int,
) -> None:
    values = totals.setdefault(key, {"loss": 0.0, "accuracy": 0.0, "count": 0.0})
    values["loss"] += loss * count
    values["accuracy"] += accuracy * count
    values["count"] += count


@torch.inference_mode()
def evaluate_causal_validation(
    *,
    step: int,
    model: PreTrainedModel,
    tokenizer: Any,
    canonical_ids: tuple[int, ...],
    validation_plan: Path,
    batch_size: int,
) -> list[dict[str, float | int | str]]:
    """Recompute pooled and component causal loss on the fixed 640-row panel."""

    dataset = SequencePlanDataset(validation_plan)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=SequenceCollator(
            tokenizer=tokenizer,
            objective="clm",
            canonical_token_ids=canonical_ids,
            mask_token_id=None,
            seed=0,
            validation_mode="diffusion",
        ),
    )
    device = next(model.parameters()).device
    totals: dict[str, dict[str, float]] = {}
    for batch in loader:
        assert_budget_reserve()
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_logits(
                model,
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                attention_mode="causal",
            )
            pooled = per_sequence_weighted_loss(
                logits,
                moved["labels"],
                moved["loss_weights"],
            )
        _accumulate(
            totals,
            "pooled",
            loss=float(pooled.loss),
            accuracy=float(pooled.accuracy),
            count=len(batch["sample_ids"]),
        )
        for component in sorted(set(batch["components"])):
            selected = torch.tensor(
                [value == component for value in batch["components"]],
                device=device,
                dtype=torch.bool,
            )
            metrics = per_sequence_weighted_loss(
                logits[selected],
                moved["labels"][selected],
                moved["loss_weights"][selected],
            )
            _accumulate(
                totals,
                component,
                loss=float(metrics.loss),
                accuracy=float(metrics.accuracy),
                count=int(selected.sum()),
            )

    expected_counts = {"pooled": 128 * len(DATA_COMPONENTS)} | {
        component.name: 128 for component in DATA_COMPONENTS
    }
    if {key: int(value["count"]) for key, value in totals.items()} != expected_counts:
        raise RuntimeError("calibration validation counts differ from the fixed panel")
    return [
        {
            "step": step,
            "component": component,
            "loss": values["loss"] / values["count"],
            "accuracy": values["accuracy"] / values["count"],
            "n_rows": int(values["count"]),
        }
        for component, values in sorted(totals.items())
    ]


def summarize_validation_gate(losses: pd.DataFrame) -> dict[str, Any]:
    """Require no end-to-end loss increase and no positive component trend."""

    required = {"step", "component", "loss", "n_rows"}
    if not required.issubset(losses.columns):
        raise ValueError(f"validation table lacks {sorted(required - set(losses.columns))}")
    checks: list[dict[str, float | int | str | bool]] = []
    expected_steps = list(CALIBRATION_CHECKPOINT_STEPS)
    for component in ("pooled", *(item.name for item in DATA_COMPONENTS)):
        selected = losses[losses["component"] == component].sort_values("step")
        if selected["step"].astype(int).tolist() != expected_steps:
            raise RuntimeError(f"{component} validation trajectory omits a checkpoint")
        steps = selected["step"].to_numpy(dtype=float)
        values = selected["loss"].to_numpy(dtype=float)
        slope = float(np.polyfit(steps, values, 1)[0])
        delta = float(values[-1] - values[0])
        checks.append(
            {
                "component": component,
                "step_0_loss": float(values[0]),
                "step_200_loss": float(values[-1]),
                "delta": delta,
                "linear_slope_per_step": slope,
                "n_rows": int(selected["n_rows"].iloc[0]),
                "passed": delta <= 0.0 and slope <= 0.0,
            }
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "criterion": "step-200 loss <= step-0 loss and linear trajectory slope <= 0",
        "checks": checks,
    }


def plot_validation_trajectories(losses: pd.DataFrame, output_path: Path) -> None:
    """Render pooled and component fixed-plan causal loss trajectories."""

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    pooled = losses[losses["component"] == "pooled"].sort_values("step")
    axes[0].plot(pooled["step"], pooled["loss"], marker="o", linewidth=1.8)
    axes[0].axhline(
        float(pooled.iloc[0]["loss"]),
        color="0.4",
        linestyle="--",
        linewidth=1,
        label="Step 0",
    )
    axes[0].set_title("Pooled fixed-plan validation")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Causal cross-entropy")
    axes[0].legend(title="Reference")

    component_rows = losses[losses["component"] != "pooled"]
    for component, rows in component_rows.groupby("component", sort=True):
        ordered = rows.sort_values("step")
        axes[1].plot(ordered["step"], ordered["loss"], marker="o", label=component)
    axes[1].set_title("Five validation components")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Causal cross-entropy")
    axes[1].legend(title="Component", fontsize="small")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_box_aspect(1)
    figure.suptitle("AdamW 1e-6 causal fine-tuning sanity check")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_stability(trace: pd.DataFrame, output_path: Path) -> None:
    """Render per-step training loss and pre-clipping gradient norm."""

    figure, axes = plt.subplots(2, 1, figsize=(6.2, 7), sharex=True, constrained_layout=True)
    axes[0].plot(
        trace["step"],
        trace["train_loss"],
        color="#4C78A8",
        alpha=0.45,
        linewidth=0.8,
        label="Per-step loss",
    )
    axes[0].plot(
        trace["step"],
        trace["train_loss"].rolling(20, min_periods=1).mean(),
        color="#1F4E79",
        linewidth=1.6,
        label="20-step mean",
    )
    axes[0].set_ylabel("Training cross-entropy")
    axes[0].set_title("Training-loss stability")
    axes[0].legend(title="Trace")

    axes[1].plot(
        trace["step"],
        trace["pre_clip_gradient_norm"],
        color="#F58518",
        linewidth=1,
        label="Pre-clip norm",
    )
    axes[1].axhline(
        CALIBRATION_MAX_GRAD_NORM,
        color="#B22222",
        linestyle="--",
        linewidth=1,
        label="Clip threshold",
    )
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Gradient L2 norm")
    axes[1].set_title("Gradient stability")
    axes[1].legend(title="Trace")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_box_aspect(1)
    figure.suptitle("AdamW 1e-6 causal fine-tuning stability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def projected_total_cost(prior_cost_usd: float) -> float:
    """Return a conservative two-hour prelaunch projection for this one stage."""

    return prior_cost_usd + CALIBRATION_MAX_INSTANCE_HOURS * LAMBDA_GH200_PRICE_PER_HOUR_USD


def run_causal_calibration(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    hf_repo_id: str,
    batch_size: int,
    seed: int,
    num_workers: int,
    offline_wandb: bool,
) -> None:
    """Train, validate, gate, and privately publish the conservative sanity arm."""

    if not torch.cuda.is_available():
        raise RuntimeError("causal calibration requires the Lambda GH200")
    if hf_repo_id not in CHECKPOINT_REPOS:
        raise ValueError(f"unexpected publication repository {hf_repo_id}")
    if batch_size != 64:
        raise ValueError(f"calibration must reuse audited batch size 64, got {batch_size}")
    assert_plan_contract(train_plan, validation_plan)
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    projection = projected_total_cost(prior_cost)
    if projection >= BUDGET_USD:
        raise RuntimeError(f"calibration projection ${projection:.2f} reaches the budget cap")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prelaunch-budget.json").write_text(
        json.dumps(
            {
                "prior_cost_usd": prior_cost,
                "maximum_instance_hours": CALIBRATION_MAX_INSTANCE_HOURS,
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
    config = AdamWCalibrationConfig()
    bundle = load_model_bundle(
        initialization="transferred",
        add_mask=False,
        attention_implementation="sdpa",
    )
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.config.use_cache = False
    module = CausalCalibrationModule(
        model=bundle.model,
        batch_size=batch_size,
        config=config,
    )
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
        group=CALIBRATION_WANDB_GROUP,
        name=CALIBRATION_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "causal-calibration", "adamw", "lr-1e-6"],
        save_dir=str(output_dir),
        offline=offline_wandb,
        log_model=False,
    )
    logger.log_hyperparams(config.to_dict())
    export = StepExportCallback(
        output_dir / "exports",
        bundle.tokenizer,
        CALIBRATION_CHECKPOINT_STEPS[1:],
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=CALIBRATION_STEPS,
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
        if trainer.global_step != CALIBRATION_STEPS:
            raise RuntimeError(
                f"calibration stopped at step {trainer.global_step}, expected {CALIBRATION_STEPS}"
            )
        if export.saved != set(CALIBRATION_CHECKPOINT_STEPS[1:]):
            raise RuntimeError(f"calibration exports differ: {sorted(export.saved)}")
        trace = pd.DataFrame(module.gradient_norm_trace)
        if trace["step"].astype(int).tolist() != list(range(CALIBRATION_STEPS)):
            raise RuntimeError("calibration gradient trace omits an optimizer step")
        trace.to_csv(output_dir / "gradient-norm-trace.csv", index=False)
        plot_training_stability(
            trace,
            output_dir / "figures" / "training-stability",
        )

        del trainer, data, module, bundle
        gc.collect()
        torch.cuda.empty_cache()

        rows: list[dict[str, float | int | str]] = []
        for step in CALIBRATION_CHECKPOINT_STEPS:
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
            rows.extend(
                evaluate_causal_validation(
                    step=step,
                    model=model,
                    tokenizer=tokenizer,
                    canonical_ids=canonical_ids,
                    validation_plan=validation_plan,
                    batch_size=batch_size,
                )
            )
            del point, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        losses = pd.DataFrame(rows)
        losses.to_csv(output_dir / "validation-loss.csv", index=False)
        gate = summarize_validation_gate(losses)
        (output_dir / "gate-summary.json").write_text(
            json.dumps(gate, indent=2) + "\n", encoding="utf-8"
        )
        plot_validation_trajectories(
            losses,
            output_dir / "figures" / "validation-trajectories",
        )

        run = logger.experiment
        run.define_metric("calibration/step")
        run.define_metric("calibration/*", step_metric="calibration/step")
        for step, selected in losses.groupby("step", sort=True):
            values = {
                f"calibration/validation/{row.component}/loss": float(row.loss)
                for row in selected.itertuples(index=False)
            }
            run.log({"calibration/step": int(step), **values})
        run.log(
            {
                "calibration/validation_table": wandb.Table(dataframe=losses),
                "calibration/validation_figure": wandb.Image(
                    str(output_dir / "figures" / "validation-trajectories.png")
                ),
                "calibration/stability_figure": wandb.Image(
                    str(output_dir / "figures" / "training-stability.png")
                ),
            }
        )
        run.summary["validation_gate_passed"] = bool(gate["passed"])
        run.summary["step_0_pooled_loss"] = float(
            losses[(losses["step"] == 0) & (losses["component"] == "pooled")]["loss"].iloc[0]
        )
        run.summary["step_200_pooled_loss"] = float(
            losses[(losses["step"] == 200) & (losses["component"] == "pooled")]["loss"].iloc[0]
        )
        artifact = wandb.Artifact(
            "dna-exp479-causal-calibration-lr1e-6",
            type="evaluation",
        )
        for filename in (
            "validation-loss.csv",
            "gate-summary.json",
            "gradient-norm-trace.csv",
            "runtime.json",
            "prelaunch-budget.json",
        ):
            artifact.add_file(str(output_dir / filename))
        artifact.add_file(str(output_dir / "figures" / "validation-trajectories.svg"))
        artifact.add_file(str(output_dir / "figures" / "training-stability.svg"))
        wandb_url = run.get_url()
        run.log_artifact(artifact)

        manifest = {
            "run_name": CALIBRATION_RUN_NAME,
            "wandb_url": wandb_url,
            "optimizer": config.to_dict(),
            "batch_size": batch_size,
            "seed": seed,
            "checkpoint_steps": list(CALIBRATION_CHECKPOINT_STEPS),
            "train_plan_sha256": plan_sha256(train_plan),
            "validation_plan_sha256": plan_sha256(validation_plan),
            "development_data": "unlabeled fixed m5.1 training and validation plans",
            "gate": gate,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        run.finish(exit_code=0)

        api = HfApi()
        assert_budget_reserve()
        api.upload_folder(
            folder_path=output_dir / "exports" / "step-0200",
            path_in_repo=CALIBRATION_FINAL_MODEL_PATH,
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message="Upload AdamW 1e-6 causal sanity checkpoint",
        )
        assert_budget_reserve()
        api.upload_folder(
            folder_path=output_dir,
            path_in_repo=CALIBRATION_REMOTE_PATH,
            repo_id=hf_repo_id,
            repo_type="model",
            allow_patterns=[
                "validation-loss.csv",
                "gate-summary.json",
                "gradient-norm-trace.csv",
                "runtime.json",
                "prelaunch-budget.json",
                "manifest.json",
                "figures/*",
            ],
            commit_message="Upload AdamW 1e-6 causal calibration evidence",
        )
    except BaseException:
        logger.experiment.finish(exit_code=1)
        raise
