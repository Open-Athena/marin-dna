"""Re-evaluate retained causal checkpoints with Marin-compatible loss normalization."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from exp479_mntp.checkpoint_audit import _loaded_from_hf, assert_plan_contract
from exp479_mntp.config import (
    DATA_COMPONENTS,
    EXPERIMENT_TAGS,
    SOURCE_Z_LOSS_WEIGHT,
    WANDB_PROJECT,
)
from exp479_mntp.data import SequenceCollator, SequencePlanDataset, plan_sha256
from exp479_mntp.masking import IGNORE_INDEX
from exp479_mntp.modeling import load_model_bundle, model_logits
from exp479_mntp.publishing import assert_budget_reserve

AUDIT_STEPS = (0, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000)
SOURCE_COMPONENTS = ("cds", "upstream", "downstream")
WANDB_ENTITY = "gonzalobenegas"
WANDB_GROUP = "dna-exp479-loss-normalization-audit"
WANDB_RUN_NAME = "dna-exp479-clm-loss-normalization-audit-seed0"
ORIGINAL_WANDB_REGION_LOSSES = {
    "cds": 0.6337850689888,
    "downstream": 0.620932400226593,
    "upstream": 0.782120406627655,
}


def checkpoint_artifact_name(step: int) -> str:
    """Return the immutable retained HF-format checkpoint artifact name."""

    if step not in AUDIT_STEPS or step == 0:
        raise ValueError(f"step {step} does not identify a retained post-update checkpoint")
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}/dna-exp479-causal-longrun-step-{step:04d}:v0"


def _new_total() -> dict[str, float]:
    return {
        "rows": 0.0,
        "selected_tokens": 0.0,
        "loss_weight_sum": 0.0,
        "weighted_ce_sum": 0.0,
        "weighted_z_loss_sum": 0.0,
        "sequence_weighted_ce_sum": 0.0,
    }


def _add_batch(
    total: dict[str, float],
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
) -> None:
    selected = labels != IGNORE_INDEX
    selected_counts = selected.sum(dim=1)
    effective_weights = torch.where(selected, loss_weights, 0.0).float()
    weight_sums = effective_weights.sum(dim=1)
    if torch.any(selected_counts == 0) or torch.any(weight_sums <= 0):
        raise ValueError("normalization audit encountered an empty validation row")

    float_logits = logits.float()
    ce = F.cross_entropy(
        float_logits.reshape(-1, float_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(labels)
    z_loss = torch.logsumexp(float_logits, dim=-1).square() * SOURCE_Z_LOSS_WEIGHT
    weighted_ce = ce * effective_weights
    weighted_z_loss = z_loss * effective_weights

    total["rows"] += float(labels.shape[0])
    total["selected_tokens"] += float(selected_counts.sum())
    total["loss_weight_sum"] += float(weight_sums.sum())
    total["weighted_ce_sum"] += float(weighted_ce.sum())
    total["weighted_z_loss_sum"] += float(weighted_z_loss.sum())
    total["sequence_weighted_ce_sum"] += float((weighted_ce.sum(dim=1) / weight_sums).sum())


def _finish_total(*, step: int, component: str, total: dict[str, float]) -> dict[str, Any]:
    return {
        "step": step,
        "component": component,
        "n_rows": int(total["rows"]),
        "selected_tokens": int(total["selected_tokens"]),
        "loss_weight_sum": total["loss_weight_sum"],
        "weight_fraction": total["loss_weight_sum"] / total["selected_tokens"],
        "legacy_count_normalized_ce": total["weighted_ce_sum"] / total["selected_tokens"],
        "sequence_balanced_ce": total["sequence_weighted_ce_sum"] / total["rows"],
        "marin_token_weighted_ce": total["weighted_ce_sum"] / total["loss_weight_sum"],
        "marin_z_loss": total["weighted_z_loss_sum"] / total["loss_weight_sum"],
        "marin_loss": (total["weighted_ce_sum"] + total["weighted_z_loss_sum"])
        / total["loss_weight_sum"],
    }


@torch.inference_mode()
def evaluate_checkpoint(
    *,
    step: int,
    model: PreTrainedModel,
    tokenizer: Any,
    canonical_ids: tuple[int, ...],
    validation_plan: Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Evaluate one causal model under legacy and corrected reducers."""

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
    totals = {"pooled": _new_total()} | {
        component.name: _new_total() for component in DATA_COMPONENTS
    }
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
        _add_batch(
            totals["pooled"],
            logits=logits,
            labels=moved["labels"],
            loss_weights=moved["loss_weights"],
        )
        for component in sorted(set(batch["components"])):
            in_component = torch.tensor(
                [value == component for value in batch["components"]],
                device=device,
                dtype=torch.bool,
            )
            _add_batch(
                totals[component],
                logits=logits[in_component],
                labels=moved["labels"][in_component],
                loss_weights=moved["loss_weights"][in_component],
            )

    expected = {"pooled": 640} | {component.name: 128 for component in DATA_COMPONENTS}
    observed = {key: int(value["rows"]) for key, value in totals.items()}
    if observed != expected:
        raise RuntimeError(f"normalization audit row counts differ: {observed}")
    return [
        _finish_total(step=step, component=component, total=total)
        for component, total in totals.items()
    ]


def macro_trajectory(component_rows: pd.DataFrame) -> pd.DataFrame:
    """Build explicit three-source and five-probe component macros."""

    expected_components = {component.name for component in DATA_COMPONENTS}
    rows: list[dict[str, Any]] = []
    for step, frame in component_rows[component_rows["component"] != "pooled"].groupby("step"):
        if set(frame["component"]) != expected_components:
            raise RuntimeError(f"step {step} lacks a validation component")
        for scope, components in (
            ("source_three", SOURCE_COMPONENTS),
            ("all_five", tuple(sorted(expected_components))),
        ):
            selected = frame[frame["component"].isin(components)]
            rows.append(
                {
                    "step": int(step),
                    "scope": scope,
                    "n_components": len(components),
                    "legacy_count_normalized_ce": float(
                        selected["legacy_count_normalized_ce"].mean()
                    ),
                    "sequence_balanced_ce": float(selected["sequence_balanced_ce"].mean()),
                    "marin_token_weighted_ce": float(selected["marin_token_weighted_ce"].mean()),
                    "marin_loss": float(selected["marin_loss"].mean()),
                }
            )
    result = pd.DataFrame(rows).sort_values(["scope", "step"]).reset_index(drop=True)
    if result.groupby("scope")["step"].apply(list).to_dict() != {
        "all_five": list(AUDIT_STEPS),
        "source_three": list(AUDIT_STEPS),
    }:
        raise RuntimeError("normalization audit macro trajectory omits a checkpoint")
    return result


def plot_corrected_trajectory(macros: pd.DataFrame, output_path: Path) -> None:
    """Plot the source-comparable fixed-panel macro through retained steps."""

    selected = macros[macros["scope"] == "source_three"].sort_values("step")
    original_region_mean = sum(ORIGINAL_WANDB_REGION_LOSSES.values()) / len(
        ORIGINAL_WANDB_REGION_LOSSES
    )
    figure, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    axis.plot(
        selected["step"],
        selected["marin_loss"],
        marker="o",
        linewidth=1.8,
        color="#4C78A8",
        label="Fixed-panel macro",
    )
    axis.axhline(
        original_region_mean,
        color="#F58518",
        linewidth=1.2,
        linestyle="--",
        label="Original run final mean",
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Marin-weighted causal loss")
    axis.set_title("Corrected validation trajectory")
    axis.grid(alpha=0.25)
    axis.legend(title="Three source datasets")
    axis.set_box_aspect(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_source_scale(macros: pd.DataFrame, output_path: Path) -> None:
    """Show how the invalid denominator changes the source-checkpoint scale."""

    step_zero = macros[macros["step"] == 0].set_index("scope")
    labels = [
        "Legacy count\nnormalization",
        "Corrected five-probe\nmacro",
        "Corrected source-three\nmacro",
    ]
    values = [
        float(step_zero.loc["all_five", "legacy_count_normalized_ce"]),
        float(step_zero.loc["all_five", "marin_loss"]),
        float(step_zero.loc["source_three", "marin_loss"]),
    ]
    figure, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    bars = axis.bar(labels, values, color=["#E45756", "#72B7B2", "#4C78A8"])
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_ylabel("Causal validation loss")
    axis.set_title("Source checkpoint under each reducer")
    axis.grid(axis="y", alpha=0.25)
    axis.set_box_aspect(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_loss_normalization_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    batch_size: int,
) -> None:
    """Evaluate the source and every retained causal checkpoint, then publish evidence."""

    if not torch.cuda.is_available():
        raise RuntimeError("loss normalization audit requires the Lambda GH200")
    if batch_size != 64:
        raise ValueError(f"normalization audit must use audited batch size 64, got {batch_size}")
    assert_plan_contract(train_plan, validation_plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run = wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=WANDB_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "loss-normalization-audit", "retained-checkpoints"],
        config={
            "validation_plan_sha256": plan_sha256(validation_plan),
            "batch_size": batch_size,
            "z_loss_weight": SOURCE_Z_LOSS_WEIGHT,
            "checkpoint_steps": list(AUDIT_STEPS),
            "checkpoint_backend": "W&B artifacts",
        },
    )
    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    try:
        for step in AUDIT_STEPS:
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
                artifact = api.artifact(checkpoint_artifact_name(step), type="model")
                root = Path(artifact.download(root=artifact_dir / "retained" / f"step-{step:04d}"))
                point = _loaded_from_hf(root / "hf", "clm")
                model = point.model
                tokenizer = point.tokenizer
                canonical_ids = point.canonical_ids
            model.to(device="cuda", dtype=torch.bfloat16).eval()
            rows.extend(
                evaluate_checkpoint(
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

        component_rows = pd.DataFrame(rows)
        macros = macro_trajectory(component_rows)
        component_path = output_dir / "component-losses.csv"
        macro_path = output_dir / "macro-trajectory.csv"
        component_rows.to_csv(component_path, index=False)
        macros.to_csv(macro_path, index=False)
        figures = output_dir / "figures"
        plot_corrected_trajectory(macros, figures / "corrected-validation-trajectory")
        plot_source_scale(macros, figures / "source-loss-scale")
        manifest = {
            "status": "completed",
            "validation_plan_sha256": plan_sha256(validation_plan),
            "checkpoint_steps": list(AUDIT_STEPS),
            "checkpoint_backend": "retained W&B model artifacts",
            "checkpoint_deletion": "not performed",
            "source_validation_scope": list(SOURCE_COMPONENTS),
            "additional_probe_scope": ["enhancer", "ncrna"],
            "original_wandb_region_losses": ORIGINAL_WANDB_REGION_LOSSES,
            "original_wandb_region_mean": sum(ORIGINAL_WANDB_REGION_LOSSES.values())
            / len(ORIGINAL_WANDB_REGION_LOSSES),
            "elapsed_seconds": time.time() - started,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        run.log(
            {
                "normalization_audit/component_losses": wandb.Table(dataframe=component_rows),
                "normalization_audit/macro_trajectory": wandb.Table(dataframe=macros),
                "normalization_audit/corrected_trajectory": wandb.Image(
                    str(figures / "corrected-validation-trajectory.png")
                ),
                "normalization_audit/source_scale": wandb.Image(
                    str(figures / "source-loss-scale.png")
                ),
            }
        )
        step_zero = macros[macros["step"] == 0].set_index("scope")
        final = macros[macros["step"] == 1_000].set_index("scope")
        run.summary["step_0_legacy_five_macro"] = float(
            step_zero.loc["all_five", "legacy_count_normalized_ce"]
        )
        run.summary["step_0_corrected_source_three_macro"] = float(
            step_zero.loc["source_three", "marin_loss"]
        )
        run.summary["step_1000_corrected_source_three_macro"] = float(
            final.loc["source_three", "marin_loss"]
        )
        result_artifact = wandb.Artifact("dna-exp479-loss-normalization-audit", type="evaluation")
        for path in (
            component_path,
            macro_path,
            manifest_path,
            figures / "corrected-validation-trajectory.svg",
            figures / "source-loss-scale.svg",
        ):
            result_artifact.add_file(str(path))
        run.log_artifact(result_artifact)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
