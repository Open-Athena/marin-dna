"""Corrected one-thousand-step transferred-MNTP trajectory for issue 479."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from matplotlib.colors import PowerNorm
from pyfaidx import Fasta
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from exp479_mntp.callbacks import BudgetGuardCallback, RuntimeMetricsCallback
from exp479_mntp.causal_longrun import (
    LONGRUN_CHECKPOINT_STEPS,
    LONGRUN_MAX_INSTANCE_HOURS,
    LONGRUN_STEPS,
    AdamWLongRunConfig,
    CausalLongRunModule,
    RetainedStepExportCallback,
    _artifact_record,
    _write_retention_manifest,
    plot_longrun_stability,
    projected_longrun_cost,
)
from exp479_mntp.checkpoint_audit import (
    ModelPoint,
    _loaded_from_hf,
    assert_plan_contract,
    evaluate_point,
)
from exp479_mntp.config import (
    BUDGET_USD,
    DATA_COMPONENTS,
    EXPERIMENT_TAGS,
    LAMBDA_GH200_PRICE_PER_HOUR_USD,
    SOURCE_Z_LOSS_WEIGHT,
    WANDB_PROJECT,
)
from exp479_mntp.data import SequenceCollator, SequencePlanDataset, plan_sha256
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.loss import per_sequence_weighted_loss
from exp479_mntp.modeling import load_model_bundle, model_logits
from exp479_mntp.module import AdaptationModule
from exp479_mntp.nucleotide_dependency import LOCI, locus_window, orientation_dependency
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate
from exp479_mntp.vep import (
    DATASETS,
    LoadedArm,
    attach_reference_windows,
    download_reference,
    load_variant_frame,
)

MNTP_WANDB_GROUP = "dna-exp479-mntp-longrun-corrected"
MNTP_RUN_NAME = "dna-exp479-transferred-mntp-adamw-1e-5-corrected-wsd1000-seed0"
MNTP_MODEL_ARTIFACT_PREFIX = "dna-exp479-mntp-longrun-corrected"
MNTP_EVALUATION_ARTIFACT = "dna-exp479-mntp-longrun-corrected-lr1e-5"
MNTP_AUPRC_STEPS = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000)
MNTP_DEPENDENCY_LOCUS = "tRNA_Arg_TCT"


class MntpLongRunModule(CausalLongRunModule):
    """Train transferred MNTP with one AdamW group and the selected schedule."""

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        batch_size: int,
        config: AdamWLongRunConfig | None = None,
    ) -> None:
        config = AdamWLongRunConfig() if config is None else config
        AdaptationModule.__init__(
            self,
            model=model,
            arm="transferred_mntp",
            batch_size=batch_size,
            train_steps=config.train_steps,
            record_gradient_norms=True,
        )
        self.longrun_config = config

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        AdaptationModule.on_save_checkpoint(self, checkpoint)
        checkpoint["exp479"]["optimizer"] = self.longrun_config.to_dict()
        checkpoint["exp479"]["mntp_longrun"] = self.longrun_config.to_dict()


def _finalize_mode(
    *,
    step: int,
    validation_mode: str,
    totals: dict[str, dict[str, float]],
) -> list[dict[str, float | int | str]]:
    expected_counts = {component.name: 128 for component in DATA_COMPONENTS}
    observed_counts = {key: int(value["count"]) for key, value in totals.items()}
    if observed_counts != expected_counts:
        raise RuntimeError(f"{validation_mode} MNTP validation counts differ: {observed_counts}")

    component_rows: list[dict[str, float | int | str]] = []
    for component in DATA_COMPONENTS:
        values = totals[component.name]
        component_rows.append(
            {
                "step": step,
                "validation_mode": validation_mode,
                "component": component.name,
                "loss": values["sequence_loss_sum"] / values["count"],
                "accuracy": values["correct_tokens"] / values["selected_tokens"],
                "n_rows": int(values["count"]),
            }
        )
    return [
        {
            "step": step,
            "validation_mode": validation_mode,
            "component": "macro",
            "loss": float(np.mean([float(row["loss"]) for row in component_rows])),
            "accuracy": float(np.mean([float(row["accuracy"]) for row in component_rows])),
            "n_rows": sum(int(row["n_rows"]) for row in component_rows),
        },
        *component_rows,
    ]


@torch.inference_mode()
def evaluate_mntp_validation(
    *,
    step: int,
    model: PreTrainedModel,
    tokenizer: Any,
    canonical_ids: tuple[int, ...],
    mask_token_id: int,
    validation_plan: Path,
    batch_size: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate pure sequence-balanced MNTP CE for both fixed mask protocols."""

    dataset = SequencePlanDataset(validation_plan)
    device = next(model.parameters()).device
    rows: list[dict[str, float | int | str]] = []
    for validation_mode in ("diffusion", "single"):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=SequenceCollator(
                tokenizer=tokenizer,
                objective="mntp",
                canonical_token_ids=canonical_ids,
                mask_token_id=mask_token_id,
                seed=0,
                validation_mode=validation_mode,
            ),
        )
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
                    attention_mode="full",
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
                    z_loss_weight=0,
                )
                count = int(selected.sum())
                values = totals.setdefault(
                    component,
                    {
                        "sequence_loss_sum": 0.0,
                        "correct_tokens": 0.0,
                        "selected_tokens": 0.0,
                        "count": 0.0,
                    },
                )
                values["sequence_loss_sum"] += float(metrics.loss) * count
                values["correct_tokens"] += float(metrics.accuracy * metrics.selected_tokens)
                values["selected_tokens"] += float(metrics.selected_tokens)
                values["count"] += count
        rows.extend(
            _finalize_mode(
                step=step,
                validation_mode=validation_mode,
                totals=totals,
            )
        )
    return rows


def summarize_mntp_trajectory(losses: pd.DataFrame) -> dict[str, Any]:
    """Require both fixed MNTP validation macros to improve end to end."""

    required = {"step", "validation_mode", "component", "loss", "n_rows"}
    if not required.issubset(losses.columns):
        raise ValueError(f"MNTP validation table lacks {sorted(required - set(losses.columns))}")
    checks: list[dict[str, float | int | str | bool]] = []
    for mode in ("diffusion", "single"):
        selected = losses[
            (losses["validation_mode"] == mode) & (losses["component"] == "macro")
        ].sort_values("step")
        if selected["step"].astype(int).tolist() != list(LONGRUN_CHECKPOINT_STEPS):
            raise RuntimeError(f"{mode} MNTP macro trajectory omits a checkpoint")
        steps = selected["step"].to_numpy(dtype=float)
        values = selected["loss"].to_numpy(dtype=float)
        delta = float(values[-1] - values[0])
        checks.append(
            {
                "validation_mode": mode,
                "step_0_loss": float(values[0]),
                "step_1000_loss": float(values[-1]),
                "delta": delta,
                "linear_slope_per_step": float(np.polyfit(steps, values, 1)[0]),
                "n_rows": int(selected["n_rows"].iloc[0]),
                "passed": delta <= 0,
            }
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "criterion": "step-1000 macro CE <= step-0 macro CE for diffusion and single-mask",
        "checks": checks,
    }


def plot_mntp_validation(losses: pd.DataFrame, output_path: Path) -> None:
    """Plot the two corrected fixed-panel MNTP validation macros."""

    macro = losses[losses["component"] == "macro"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = (
        ("diffusion", "Diffusion-mask validation"),
        ("single", "Single-mask validation"),
    )
    for axis, (mode, title) in zip(axes, panels, strict=True):
        selected = macro[macro["validation_mode"] == mode].sort_values("step")
        axis.plot(
            selected["step"],
            selected["loss"],
            marker="o",
            color="#E45756",
            linewidth=1.8,
            label="Five-component macro",
        )
        axis.axhline(
            float(selected.iloc[0]["loss"]),
            color="0.4",
            linestyle="--",
            linewidth=1,
            label="Step 0",
        )
        axis.axvline(100, color="#999999", linestyle=":", linewidth=1)
        axis.axvline(800, color="#999999", linestyle=":", linewidth=1)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Validation cross-entropy")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(title="Trajectory")
        axis.set_box_aspect(1)
    figure.suptitle("Transferred MNTP: AdamW 1e-5 with corrected loss")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mntp_auprc(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot registered odd-autosome/X FWD+RC AUPRC trajectories."""

    datasets = ("mendelian_traits", "complex_traits", "sge")
    titles = ("Mendelian traits", "Complex traits", "SGE")
    selected = metrics[metrics["orientation"] == "protocol_fwd_rc"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, dataset, title in zip(axes, datasets, titles, strict=True):
        cell = selected[selected["dataset"] == dataset].sort_values("step")
        axis.errorbar(
            cell["step"],
            cell["auprc"],
            yerr=cell["se"],
            color="#E45756",
            marker="o",
            linewidth=1.5,
            markersize=4,
            capsize=2,
            label="Transferred MNTP",
        )
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("AUPRC")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_box_aspect(1)
    axes[0].legend(title="Registered FWD+RC")
    figure.suptitle("Corrected transferred-MNTP odd-autosome/X VEP trajectory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _dependency_summary(matrix: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for region, indices in (
        ("past_context", np.triu_indices(matrix.shape[0], k=1)),
        ("future_context", np.tril_indices(matrix.shape[0], k=-1)),
    ):
        values = matrix[indices]
        rows.append(
            {
                "locus": MNTP_DEPENDENCY_LOCUS,
                "region": region,
                "mean_dependency": float(values.mean()),
                "p95_dependency": float(np.quantile(values, 0.95)),
                "maximum_dependency": float(values.max()),
                "nonzero_fraction": float(np.mean(values > 0)),
                "n_position_pairs": len(values),
            }
        )
    return pd.DataFrame(rows)


def plot_mntp_dependency(matrix: np.ndarray, output_path: Path) -> None:
    """Plot one final-checkpoint directed dependency map."""

    maximum = float(matrix.max())
    if maximum <= 0:
        raise RuntimeError("final MNTP dependency map is zero")
    figure, axis = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    image = axis.imshow(
        matrix,
        origin="lower",
        cmap="viridis",
        norm=PowerNorm(gamma=0.45, vmin=0, vmax=maximum),
        interpolation="nearest",
        rasterized=True,
    )
    axis.plot((0, matrix.shape[0] - 1), (0, matrix.shape[0] - 1), color="white", lw=0.5)
    axis.set_xlabel("Readout position")
    axis.set_ylabel("Substitution position")
    axis.set_title("tRNA-Arg-TCT directed dependency")
    figure.colorbar(image, ax=axis, label="L∞ change in A/C/G/T log probability")
    figure.suptitle("Corrected transferred MNTP at step 1,000")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _load_trajectory_point(
    *,
    step: int,
    output_dir: Path,
) -> tuple[LoadedArm, ModelPoint]:
    if step == 0:
        bundle = load_model_bundle(
            initialization="transferred",
            add_mask=True,
            attention_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        loaded = LoadedArm(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            canonical_ids=bundle.canonical_token_ids,
            mask_token_id=bundle.mask_token_id,
        )
        kind = "no_adaptation"
    else:
        loaded = _loaded_from_hf(output_dir / "exports" / f"step-{step:04d}", "mntp")
        kind = "hf"
    point = ModelPoint(
        point_id=f"mntp-corrected-step{step:04d}",
        arm="transferred_mntp",
        step=step,
        objective="mntp",
        kind=kind,
        plot_series="Transferred MNTP AdamW 1e-5",
        path=None if step == 0 else output_dir / "exports" / f"step-{step:04d}",
    )
    return loaded, point


def run_mntp_longrun(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    batch_size: int,
    seed: int,
    num_workers: int,
    offline_wandb: bool,
    vep_batch_size: int,
    dependency_batch_size: int,
    n_bootstrap: int,
) -> None:
    """Train, retain, and evaluate the selected corrected transferred-MNTP run."""

    if not torch.cuda.is_available():
        raise RuntimeError("MNTP long run requires the Lambda GH200")
    if batch_size != 64:
        raise ValueError(f"MNTP long run must reuse audited batch size 64, got {batch_size}")
    if vep_batch_size <= 0:
        raise ValueError("VEP batch size must be positive")
    if dependency_batch_size <= 1:
        raise ValueError("dependency batch size must leave room for a baseline")
    assert_plan_contract(train_plan, validation_plan)

    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    projection = projected_longrun_cost(prior_cost)
    if projection >= BUDGET_USD:
        raise RuntimeError(f"MNTP long-run projection ${projection:.2f} reaches the budget cap")
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
        add_mask=True,
        attention_implementation="sdpa",
    )
    if bundle.mask_token_id is None:
        raise RuntimeError("MNTP model lacks a MASK token")
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.config.use_cache = False
    module = MntpLongRunModule(model=bundle.model, batch_size=batch_size, config=config)
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective="mntp",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )
    logger = WandbLogger(
        project=WANDB_PROJECT,
        group=MNTP_WANDB_GROUP,
        name=MNTP_RUN_NAME,
        tags=[
            *EXPERIMENT_TAGS,
            "transferred-mntp",
            "corrected-loss",
            "adamw",
            "lr-1e-5",
            "wandb-retained",
        ],
        save_dir=str(output_dir),
        offline=offline_wandb,
        log_model=False,
    )
    logger.log_hyperparams(
        config.to_dict()
        | {
            "objective": "mntp",
            "attention_mode": "full",
            "mask_initialization": "independent A/C/G/T row means",
        }
    )
    run = logger.experiment
    export = RetainedStepExportCallback(
        output_dir / "exports",
        bundle.tokenizer,
        LONGRUN_CHECKPOINT_STEPS[1:],
        run,
        artifact_prefix=MNTP_MODEL_ARTIFACT_PREFIX,
        objective="mntp",
        artifact_objective="mntp_sequence_balanced_corrected_repeat_weight",
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
                f"MNTP long run stopped at step {trainer.global_step}, expected {LONGRUN_STEPS}"
            )
        if export.saved != set(LONGRUN_CHECKPOINT_STEPS[1:]):
            raise RuntimeError(f"MNTP long-run exports differ: {sorted(export.saved)}")

        trace = pd.DataFrame(module.gradient_norm_trace)
        if trace["step"].astype(int).tolist() != list(range(LONGRUN_STEPS)):
            raise RuntimeError("MNTP long-run gradient trace omits an optimizer step")
        trace.to_csv(output_dir / "gradient-norm-trace.csv", index=False)
        plot_longrun_stability(
            trace,
            output_dir / "figures" / "training-stability",
            title="Corrected transferred-MNTP AdamW 1e-5 training stability",
        )

        final_checkpoint = output_dir / "checkpoints" / "step-1000.ckpt"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(final_checkpoint)
        assert_budget_reserve()
        checkpoint_artifact = wandb.Artifact(
            f"{MNTP_MODEL_ARTIFACT_PREFIX}-step-1000-full",
            type="model",
            metadata={
                "optimizer_step": LONGRUN_STEPS,
                "objective": "mntp_sequence_balanced_corrected_repeat_weight",
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

        reference = download_reference(artifact_dir / "reference")
        frames = {
            spec.name: attach_reference_windows(load_variant_frame(spec), reference)
            for spec in DATASETS
        }
        validation_rows: list[dict[str, float | int | str]] = []
        auprc_rows: list[dict[str, object]] = []
        dependency_summary: pd.DataFrame | None = None
        dependency_npz = output_dir / "nucleotide-dependency.npz"
        with Fasta(reference, as_raw=True, rebuild=False) as genome:
            locus = next(item for item in LOCI if item.name == MNTP_DEPENDENCY_LOCUS)
            dependency_sequence, dependency_start = locus_window(genome, locus)

        for step in LONGRUN_CHECKPOINT_STEPS:
            assert_budget_reserve()
            loaded, point = _load_trajectory_point(step=step, output_dir=output_dir)
            loaded.model.to(device="cuda", dtype=torch.bfloat16).eval()
            if loaded.mask_token_id is None:
                raise RuntimeError(f"MNTP checkpoint {step} lacks MASK metadata")
            validation_rows.extend(
                evaluate_mntp_validation(
                    step=step,
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    canonical_ids=loaded.canonical_ids,
                    mask_token_id=loaded.mask_token_id,
                    validation_plan=validation_plan,
                    batch_size=batch_size,
                )
            )
            if step in MNTP_AUPRC_STEPS:
                auprc_rows.extend(
                    evaluate_point(
                        point,
                        loaded,
                        frames,
                        output_dir / "private-auprc-scores",
                        batch_size=vep_batch_size,
                        n_bootstrap=n_bootstrap,
                    )
                )
            if step == LONGRUN_STEPS:
                dependency = orientation_dependency(
                    loaded,
                    dependency_sequence,
                    batch_size=dependency_batch_size,
                    attention_mode="full",
                )
                np.savez_compressed(dependency_npz, directed=dependency)
                dependency_summary = _dependency_summary(dependency)
                dependency_summary.to_csv(
                    output_dir / "nucleotide-dependency-summary.csv",
                    index=False,
                )
                plot_mntp_dependency(
                    dependency,
                    output_dir / "figures" / "nucleotide-dependency",
                )
            del loaded
            gc.collect()
            torch.cuda.empty_cache()

        losses = pd.DataFrame(validation_rows)
        losses.to_csv(output_dir / "validation-loss.csv", index=False)
        gate = summarize_mntp_trajectory(losses)
        gate_path = output_dir / "gate-summary.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        plot_mntp_validation(losses, output_dir / "figures" / "validation-trajectory")

        metrics = pd.DataFrame(auprc_rows)
        metrics.to_csv(output_dir / "checkpoint-auprc.csv", index=False)
        plot_mntp_auprc(metrics, output_dir / "figures" / "auprc-trajectory")
        if dependency_summary is None:
            raise RuntimeError("final-checkpoint dependency analysis did not run")

        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest = {
            "run_name": MNTP_RUN_NAME,
            "wandb_url": run.get_url(),
            "optimizer": config.to_dict(),
            "batch_size": batch_size,
            "seed": seed,
            "checkpoint_steps": list(LONGRUN_CHECKPOINT_STEPS),
            "auprc_steps": list(MNTP_AUPRC_STEPS),
            "train_plan_sha256": plan_sha256(train_plan),
            "validation_plan_sha256": plan_sha256(validation_plan),
            "training_objective": (
                "sequence-balanced effective-weight mean of MNTP CE plus source z-loss"
            ),
            "training_z_loss_weight": SOURCE_Z_LOSS_WEIGHT,
            "attention_mode": "full",
            "validation_objective": (
                "pure sequence-balanced MNTP CE with one repeat-weight application"
            ),
            "validation_scope": (
                "equal macro of five 128-row components for diffusion and single-mask"
            ),
            "development_split": "odd autosomes and X only",
            "vep_orientation": "registered FWD+RC",
            "dependency_locus": MNTP_DEPENDENCY_LOCUS,
            "dependency_context_start_zero_based": dependency_start,
            "dependency_context_end_zero_based_exclusive": (
                dependency_start + len(dependency_sequence)
            ),
            "model_artifact_prefix": MNTP_MODEL_ARTIFACT_PREFIX,
            "checkpoint_retention": "W&B model artifacts listed in retention-manifest.json",
            "checkpoint_deletion": "not performed",
            "gate": gate,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        macro = losses[losses["component"] == "macro"].sort_values(["validation_mode", "step"])
        run.define_metric("mntp_longrun_corrected/step")
        run.define_metric(
            "mntp_longrun_corrected/*",
            step_metric="mntp_longrun_corrected/step",
        )
        for step in LONGRUN_CHECKPOINT_STEPS:
            payload: dict[str, float | int] = {"mntp_longrun_corrected/step": step}
            for mode in ("diffusion", "single"):
                selected = macro[(macro["step"] == step) & (macro["validation_mode"] == mode)]
                payload[f"mntp_longrun_corrected/validation/{mode}_macro_ce"] = float(
                    selected.iloc[0]["loss"]
                )
            run.log(payload)
        run.log(
            {
                "mntp_longrun_corrected/validation_table": wandb.Table(dataframe=losses),
                "mntp_longrun_corrected/auprc_table": wandb.Table(dataframe=metrics),
                "mntp_longrun_corrected/dependency_summary": wandb.Table(
                    dataframe=dependency_summary
                ),
                "mntp_longrun_corrected/validation_figure": wandb.Image(
                    str(output_dir / "figures" / "validation-trajectory.png")
                ),
                "mntp_longrun_corrected/stability_figure": wandb.Image(
                    str(output_dir / "figures" / "training-stability.png")
                ),
                "mntp_longrun_corrected/auprc_figure": wandb.Image(
                    str(output_dir / "figures" / "auprc-trajectory.png")
                ),
                "mntp_longrun_corrected/dependency_figure": wandb.Image(
                    str(output_dir / "figures" / "nucleotide-dependency.png")
                ),
            }
        )
        run.summary["validation_gate_passed"] = bool(gate["passed"])
        run.summary["checkpoint_retention"] = "W&B model artifacts"
        evaluation_artifact = wandb.Artifact(MNTP_EVALUATION_ARTIFACT, type="evaluation")
        for path in (
            output_dir / "validation-loss.csv",
            output_dir / "checkpoint-auprc.csv",
            output_dir / "nucleotide-dependency-summary.csv",
            dependency_npz,
            gate_path,
            output_dir / "gradient-norm-trace.csv",
            output_dir / "runtime.json",
            budget_path,
            retention_path,
            manifest_path,
            cost_path,
            output_dir / "figures" / "validation-trajectory.svg",
            output_dir / "figures" / "training-stability.svg",
            output_dir / "figures" / "auprc-trajectory.svg",
            output_dir / "figures" / "nucleotide-dependency.svg",
        ):
            evaluation_artifact.add_file(str(path))
        logged_evaluation = run.log_artifact(evaluation_artifact)
        logged_evaluation.wait()
        run.finish(exit_code=0)
    except BaseException:
        logger.experiment.finish(exit_code=1)
        raise
