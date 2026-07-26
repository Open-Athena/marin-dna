#!/usr/bin/env python3
"""Plot issue #402 large-batch validation loss by step and processed tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import polars as pl
import seaborn as sns
import wandb

LOSS_KEY = "eval/datasets/dna-exp402-rag-tokenized/loss"
RUNS = {
    "46M": "gonzalobenegas/marin/dna-exp402-rag-h640-p46M-B2M-30K-scratch",
    "104M": "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-B2M-30K-scratch",
}
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}
TOKENS_PER_UPDATE = 1_024 * 2_048
WARMUP_END = 3_000
STABLE_END = 24_000
FINAL_STEP = 29_999
EXPECTED_STEPS = (*range(1_000, 30_000, 1_000), FINAL_STEP)
AXIS_ORDER = ["Training step", "Processed tokens (billions)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_large_batch_validation_loss"),
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Render the available prefix instead of requiring completed runs.",
    )
    return parser.parse_args()


def load_validation_losses(
    api: wandb.Api, *, allow_running: bool = False
) -> pl.DataFrame:
    """Fetch exact scheduled validation points from both production runs."""
    rows: list[dict[str, object]] = []
    for model, run_path in RUNS.items():
        run = api.run(run_path)
        if not allow_running:
            assert run.state == "finished", (model, run.state)
        else:
            assert run.state in {"running", "finished"}, (model, run.state)
        for row in run.scan_history(keys=["global_step", LOSS_KEY], page_size=1_000):
            if row.get("global_step") is None or row.get(LOSS_KEY) is None:
                continue
            step = int(row["global_step"])
            assert step in EXPECTED_STEPS, (model, step)
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "validation_loss": float(row[LOSS_KEY]),
                    "run_state": run.state,
                }
            )
    assert rows
    raw = pl.DataFrame(rows)
    consistency = raw.group_by("model", "step").agg(
        pl.col("validation_loss").n_unique().alias("n_unique_losses"),
        pl.col("run_state").n_unique().alias("n_unique_states"),
    )
    assert consistency.filter(
        (pl.col("n_unique_losses") != 1) | (pl.col("n_unique_states") != 1)
    ).is_empty()
    losses = (
        raw.unique(["model", "step", "validation_loss", "run_state"])
        .with_columns(
            (pl.col("step") + 1).alias("completed_updates"),
            ((pl.col("step") + 1) * TOKENS_PER_UPDATE).alias("training_tokens"),
            ((pl.col("step") + 1) * TOKENS_PER_UPDATE / 1e9).alias(
                "training_tokens_billions"
            ),
            pl.when(pl.col("step") < WARMUP_END)
            .then(pl.lit("warmup"))
            .when(pl.col("step") < STABLE_END)
            .then(pl.lit("stable"))
            .otherwise(pl.lit("decay"))
            .alias("schedule_phase"),
        )
        .sort("model", "step")
    )
    assert losses.filter(
        ~pl.col("validation_loss").is_finite() | (pl.col("validation_loss") <= 0)
    ).is_empty()
    for model in MODEL_ORDER:
        model_steps = losses.filter(pl.col("model") == model)["step"].to_list()
        assert model_steps == sorted(set(model_steps))
        if not allow_running:
            assert model_steps == list(EXPECTED_STEPS), (model, model_steps)
    assert (FINAL_STEP + 1) * TOKENS_PER_UPDATE == 62_914_560_000
    return losses


def _format_step(value: float, _position: int) -> str:
    return f"{value / 1_000:g}k"


def _format_billions(value: float, _position: int) -> str:
    return f"{value:g}B"


def plot_validation_losses(losses: pl.DataFrame, output_dir: Path) -> None:
    """Render identical loss values on step and token x-axes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    losses.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    plot_data = pl.concat(
        [
            losses.with_columns(
                pl.col("step").cast(pl.Float64).alias("progress"),
                pl.lit(AXIS_ORDER[0]).alias("progress_axis"),
            ),
            losses.with_columns(
                pl.col("training_tokens_billions").alias("progress"),
                pl.lit(AXIS_ORDER[1]).alias("progress_axis"),
            ),
        ]
    ).to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=plot_data,
        x="progress",
        y="validation_loss",
        hue="model",
        hue_order=MODEL_ORDER,
        style="model",
        style_order=MODEL_ORDER,
        col="progress_axis",
        col_order=AXIS_ORDER,
        kind="line",
        markers=True,
        dashes=False,
        palette=MODEL_COLORS,
        height=4.8,
        aspect=1.12,
        facet_kws={"sharex": False, "sharey": True},
    )
    grid.set_axis_labels("", "Validation loss")
    for axis_name, axis in grid.axes_dict.items():
        axis.set_title(axis_name)
        if axis_name == AXIS_ORDER[0]:
            axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
            boundaries = (
                (WARMUP_END, "warmup→stable"),
                (STABLE_END, "stable→decay"),
            )
        else:
            axis.xaxis.set_major_formatter(FuncFormatter(_format_billions))
            boundaries = (
                (WARMUP_END * TOKENS_PER_UPDATE / 1e9, "warmup→stable"),
                (STABLE_END * TOKENS_PER_UPDATE / 1e9, "stable→decay"),
            )
        for value, label in boundaries:
            axis.axvline(value, color="#737373", linewidth=1, linestyle=":")
            axis.annotate(
                label,
                xy=(value, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(3, 0),
                textcoords="offset points",
                rotation=90,
                va="top",
                fontsize=8,
                color="#555555",
            )
        for model in MODEL_ORDER:
            model_rows = losses.filter(pl.col("model") == model)
            if model_rows.is_empty():
                continue
            latest = model_rows.tail(1).row(0, named=True)
            x_value = (
                float(latest["step"])
                if axis_name == AXIS_ORDER[0]
                else float(latest["training_tokens_billions"])
            )
            axis.annotate(
                f"{model}: {latest['validation_loss']:.4f}",
                (x_value, latest["validation_loss"]),
                xytext=(-8, 8 if model == "46M" else -15),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color=MODEL_COLORS[model],
            )
    complete = set(losses["run_state"]) == {"finished"}
    qualifier = "complete" if complete else "live partial"
    grid.figure.suptitle(f"Large-batch ortholog-RAG validation loss ({qualifier})")
    grid.figure.text(
        0.5,
        0.012,
        "Fixed 2,048-document validation set; 2,097,152 tokens per update. "
        "Checkpoint indices are zero-based, so processed tokens use step + 1.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.82, bottom=0.18, wspace=0.16)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    losses = load_validation_losses(wandb.Api(), allow_running=args.allow_running)
    plot_validation_losses(losses, args.output_dir)
    print(losses)


if __name__ == "__main__":
    main()
