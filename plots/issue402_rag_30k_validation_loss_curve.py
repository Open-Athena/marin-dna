#!/usr/bin/env python3
"""Plot issue #402's two complete 30k scratch validation-loss trajectories."""

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
    "46M": "gonzalobenegas/marin/dna-exp402-rag-h640-p46M-30K-scratch",
    "104M": "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-30K-scratch",
}
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}
TOKENS_PER_UPDATE = 64 * 2_048
WARMUP_END = 3_000
STABLE_END = 24_000
FINAL_STEP = 29_999
EXPECTED_STEPS = (*range(1_000, 30_000, 1_000), FINAL_STEP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_30k_validation_loss_curve"),
    )
    return parser.parse_args()


def load_validation_losses(api: wandb.Api) -> pl.DataFrame:
    """Fetch and validate every scheduled 30k validation point."""
    rows: list[dict[str, object]] = []
    for model, run_path in RUNS.items():
        run = api.run(run_path)
        assert run.state == "finished", (model, run.state)
        for row in run.scan_history(keys=["global_step", LOSS_KEY], page_size=1_000):
            if row.get("global_step") is None or row.get(LOSS_KEY) is None:
                continue
            rows.append(
                {
                    "model": model,
                    "step": int(row["global_step"]),
                    "validation_loss": float(row[LOSS_KEY]),
                }
            )
    raw = pl.DataFrame(rows)
    consistency = raw.group_by("model", "step").agg(
        pl.col("validation_loss").n_unique().alias("n_unique_losses")
    )
    assert consistency.filter(pl.col("n_unique_losses") != 1).is_empty()
    losses = (
        raw.unique(["model", "step", "validation_loss"])
        .with_columns(
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
    assert losses.height == len(RUNS) * len(EXPECTED_STEPS)
    for model in MODEL_ORDER:
        model_rows = losses.filter(pl.col("model") == model)
        assert model_rows["step"].to_list() == list(EXPECTED_STEPS)
    assert losses.filter(
        ~pl.col("validation_loss").is_finite() | (pl.col("validation_loss") <= 0)
    ).is_empty()
    assert (FINAL_STEP + 1) * TOKENS_PER_UPDATE == 3_932_160_000
    return losses


def _format_step(value: float, _position: int) -> str:
    return f"{value / 1_000:g}k"


def plot_validation_losses(losses: pl.DataFrame, output_dir: Path) -> None:
    """Render direct loss-versus-step curves with WSD phase boundaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    losses.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    frame = losses.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="step",
        y="validation_loss",
        hue="model",
        hue_order=MODEL_ORDER,
        style="model",
        style_order=MODEL_ORDER,
        kind="line",
        markers=True,
        dashes=False,
        palette=MODEL_COLORS,
        height=5.2,
        aspect=1.65,
    )
    axis = grid.ax
    for step, label in ((WARMUP_END, "warmup→stable"), (STABLE_END, "stable→decay")):
        axis.axvline(step, color="#737373", linewidth=1, linestyle=":")
        axis.annotate(
            label,
            xy=(step, 0.615),
            xytext=(3, 0),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=9,
            color="#555555",
        )
    for model in MODEL_ORDER:
        final = losses.filter(
            (pl.col("model") == model) & (pl.col("step") == FINAL_STEP)
        ).row(0, named=True)
        axis.annotate(
            f"{model}: {final['validation_loss']:.4f}",
            (final["step"], final["validation_loss"]),
            xytext=(-8, 8 if model == "46M" else -15),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            color=MODEL_COLORS[model],
        )
    axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
    grid.set_axis_labels("Training step", "Validation loss")
    grid.figure.suptitle("Validation loss keeps falling through the 30k-step endpoint")
    grid.figure.text(
        0.5,
        0.015,
        "Fixed 2,048-document validation set (4.19M tokens), evaluated every "
        "1,000 steps. Final step 29,999 is 30,000 updates / 3.932B tokens.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.84, bottom=0.18)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    losses = load_validation_losses(wandb.Api())
    plot_validation_losses(losses, args.output_dir)
    print(losses)


if __name__ == "__main__":
    main()
