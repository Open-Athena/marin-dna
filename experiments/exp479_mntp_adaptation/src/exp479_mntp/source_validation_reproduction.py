"""Reproduce the original m5.1 validation metrics on all source rows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset

from exp479_mntp.config import (
    DATA_COMPONENTS,
    EXPERIMENT_TAGS,
    SOURCE_Z_LOSS_WEIGHT,
    WANDB_PROJECT,
)
from exp479_mntp.data import SequenceCollator
from exp479_mntp.masking import IGNORE_INDEX
from exp479_mntp.modeling import load_model_bundle, model_logits
from exp479_mntp.publishing import assert_budget_reserve

EXPECTED_ROWS_PER_COMPONENT = 16_384
SOURCE_COMPONENTS = tuple(DATA_COMPONENTS[:3])
SLICE_WEIGHTS = ("default", "functional", "nonfunctional")
PARITY_TOLERANCE = 2e-3
WANDB_GROUP = "dna-exp479-source-validation-reproduction"
WANDB_RUN_NAME = "dna-exp479-source-validation-reproduction"
ORIGINAL_WANDB_LOSSES = {
    "cds/default": 0.6337850689888,
    "cds/functional": 0.644463062286377,
    "cds/nonfunctional": 1.2025692462921145,
    "downstream/default": 0.620932400226593,
    "downstream/functional": 0.6525456309318542,
    "downstream/nonfunctional": 1.1660317182540894,
    "upstream/default": 0.782120406627655,
    "upstream/functional": 0.8157918453216553,
    "upstream/nonfunctional": 1.2338632345199585,
}
ORIGINAL_WANDB_MACRO = 0.8613447546958923


class ComponentValidationDataset(Dataset[dict[str, Any]]):
    """Expose every row of one pinned public source validation split."""

    def __init__(self, component_index: int) -> None:
        from datasets import load_dataset

        component = SOURCE_COMPONENTS[component_index]
        self.component = component
        self.dataset = load_dataset(
            component.validation_repo,
            split="validation",
            revision=component.validation_revision,
        )
        if len(self.dataset) != EXPECTED_ROWS_PER_COMPONENT:
            raise RuntimeError(
                f"{component.name} has {len(self.dataset)} rows, "
                f"expected {EXPECTED_ROWS_PER_COMPONENT}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence = str(self.dataset[index][self.component.validation_text_key])
        if len(sequence) != 255:
            raise ValueError(f"{self.component.name} row {index} has {len(sequence)} bases")
        return {"sample_id": index, "component": self.component.name, "sequence": sequence}


def _new_total() -> dict[str, float]:
    return {"weighted_loss_sum": 0.0, "loss_weight_sum": 0.0, "selected_tokens": 0.0}


def _add_weighted_loss(
    total: dict[str, float],
    *,
    per_token_loss: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    selected = labels != IGNORE_INDEX
    effective_weights = torch.where(selected, weights, 0.0).float()
    total["weighted_loss_sum"] += float((per_token_loss * effective_weights).sum())
    total["loss_weight_sum"] += float(effective_weights.sum())
    total["selected_tokens"] += float(selected.sum())


@torch.inference_mode()
def evaluate_source_component(
    *,
    component_index: int,
    model: Any,
    tokenizer: Any,
    canonical_ids: tuple[int, ...],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Evaluate all rows from one original source validation dataset."""

    dataset = ComponentValidationDataset(component_index)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
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
    totals = {name: _new_total() for name in SLICE_WEIGHTS}
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
        float_logits = logits.float()
        labels = moved["labels"]
        ce = F.cross_entropy(
            float_logits.reshape(-1, float_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape_as(labels)
        per_token_loss = ce + torch.logsumexp(float_logits, dim=-1).square() * SOURCE_Z_LOSS_WEIGHT
        default_weights = moved["loss_weights"]
        weights = {
            "default": default_weights,
            "functional": (default_weights > 0.5).float(),
            "nonfunctional": ((default_weights > 0) & (default_weights < 0.5)).float(),
        }
        for slice_name, slice_weights in weights.items():
            _add_weighted_loss(
                totals[slice_name],
                per_token_loss=per_token_loss,
                labels=labels,
                weights=slice_weights,
            )

    component = SOURCE_COMPONENTS[component_index].name
    rows = []
    for slice_name, total in totals.items():
        if total["loss_weight_sum"] <= 0:
            raise RuntimeError(f"{component}/{slice_name} has zero effective weight")
        metric = f"{component}/{slice_name}"
        reproduced = total["weighted_loss_sum"] / total["loss_weight_sum"]
        original = ORIGINAL_WANDB_LOSSES[metric]
        rows.append(
            {
                "component": component,
                "slice": slice_name,
                "metric": metric,
                "n_rows": len(dataset),
                "selected_tokens": int(total["selected_tokens"]),
                "loss_weight_sum": total["loss_weight_sum"],
                "reproduced_loss": reproduced,
                "original_wandb_loss": original,
                "delta": reproduced - original,
                "absolute_delta": abs(reproduced - original),
            }
        )
    return rows


def summarize_source_parity(frame: pd.DataFrame) -> dict[str, Any]:
    """Compare nine reproduced slices and their macro with original W&B."""

    if set(frame["metric"]) != set(ORIGINAL_WANDB_LOSSES):
        raise RuntimeError("source validation reproduction lacks an original W&B metric")
    if set(frame["n_rows"]) != {EXPECTED_ROWS_PER_COMPONENT}:
        raise RuntimeError("source validation reproduction row counts changed")
    reproduced_macro = float(frame["reproduced_loss"].mean())
    macro_delta = reproduced_macro - ORIGINAL_WANDB_MACRO
    maximum_absolute_delta = float(frame["absolute_delta"].max())
    return {
        "passed": maximum_absolute_delta <= PARITY_TOLERANCE
        and abs(macro_delta) <= PARITY_TOLERANCE,
        "tolerance": PARITY_TOLERANCE,
        "reproduced_macro": reproduced_macro,
        "original_wandb_macro": ORIGINAL_WANDB_MACRO,
        "macro_delta": macro_delta,
        "maximum_slice_absolute_delta": maximum_absolute_delta,
        "n_metrics": len(frame),
        "total_model_rows_evaluated": EXPECTED_ROWS_PER_COMPONENT * len(SOURCE_COMPONENTS),
    }


def plot_source_parity(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot reproduced against original W&B loss for all nine slices."""

    figure, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    colors = {"default": "#4C78A8", "functional": "#F58518", "nonfunctional": "#54A24B"}
    for slice_name in SLICE_WEIGHTS:
        selected = frame[frame["slice"] == slice_name]
        axis.scatter(
            selected["original_wandb_loss"],
            selected["reproduced_loss"],
            s=55,
            color=colors[slice_name],
            label=slice_name.replace("nonfunctional", "Lowercase-only")
            .replace("functional", "Uppercase-only")
            .replace("default", "Repeat-weighted"),
        )
        for row in selected.itertuples(index=False):
            axis.annotate(
                row.component,
                (row.original_wandb_loss, row.reproduced_loss),
                xytext=(4, 4),
                textcoords="offset points",
            )
    low = min(frame["original_wandb_loss"].min(), frame["reproduced_loss"].min()) - 0.03
    high = max(frame["original_wandb_loss"].max(), frame["reproduced_loss"].max()) + 0.03
    axis.plot([low, high], [low, high], color="0.4", linestyle="--", linewidth=1)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Original W&B loss")
    axis.set_ylabel("Reproduced loss")
    axis.set_title("Full source-validation parity")
    axis.grid(alpha=0.25)
    axis.legend(title="Validation slice")
    axis.set_box_aspect(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_source_validation_reproduction(*, output_dir: Path, batch_size: int) -> None:
    """Evaluate the released source on every original validation row."""

    if not torch.cuda.is_available():
        raise RuntimeError("source validation reproduction requires the Lambda GH200")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run = wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=WANDB_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "source-validation", "loss-parity", "full-dataset"],
        config={
            "batch_size": batch_size,
            "rows_per_component": EXPECTED_ROWS_PER_COMPONENT,
            "z_loss_weight": SOURCE_Z_LOSS_WEIGHT,
            "parity_tolerance": PARITY_TOLERANCE,
        },
    )
    try:
        bundle = load_model_bundle(
            initialization="transferred",
            add_mask=False,
            attention_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        bundle.model.to(device="cuda", dtype=torch.bfloat16).eval()
        rows = []
        for component_index in range(len(SOURCE_COMPONENTS)):
            rows.extend(
                evaluate_source_component(
                    component_index=component_index,
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    canonical_ids=bundle.canonical_token_ids,
                    batch_size=batch_size,
                )
            )
        frame = pd.DataFrame(rows)
        summary = summarize_source_parity(frame)
        frame_path = output_dir / "source-validation-parity.csv"
        summary_path = output_dir / "summary.json"
        manifest_path = output_dir / "manifest.json"
        frame.to_csv(frame_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        plot_source_parity(frame, output_dir / "source-validation-parity")
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "source_components": [component.name for component in SOURCE_COMPONENTS],
                    "validation_slices": list(SLICE_WEIGHTS),
                    "rows_per_component": EXPECTED_ROWS_PER_COMPONENT,
                    "total_model_rows_evaluated": EXPECTED_ROWS_PER_COMPONENT
                    * len(SOURCE_COMPONENTS),
                    "elapsed_seconds": time.time() - started,
                    "checkpoint_deletion": "not performed",
                    "checkpoint_upload": "not performed",
                    "summary": summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run.log(
            {
                "source_validation/parity_table": wandb.Table(dataframe=frame),
                "source_validation/parity_figure": wandb.Image(
                    str(output_dir / "source-validation-parity.png")
                ),
            }
        )
        for key, value in summary.items():
            run.summary[key] = value
        artifact = wandb.Artifact("dna-exp479-source-validation-reproduction", type="evaluation")
        for path in (
            frame_path,
            summary_path,
            manifest_path,
            output_dir / "source-validation-parity.svg",
        ):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        if not summary["passed"]:
            raise RuntimeError(
                "full source-validation reproduction exceeds the "
                f"{PARITY_TOLERANCE} parity tolerance: {summary}"
            )
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
