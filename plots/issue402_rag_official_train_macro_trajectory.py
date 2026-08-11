#!/usr/bin/env python3
"""Plot official train-split Mendelian macro AUPRC for issue #402 RAG models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.ticker import FuncFormatter

RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
MODEL_CHECKPOINTS = {
    "46M": {
        10_000: "exp402-rag-h640-p46m-b2m-step-10000",
        20_000: "exp402-rag-h640-p46m-b2m-step-20000",
        30_000: "exp402-rag-h640-p46m-b2m-step-29999",
    },
    "104M": {
        10_000: "exp402-rag-h768-p104m-b2m-step-10000",
        20_000: "exp402-rag-h768-p104m-b2m-step-20000",
        30_000: "exp402-rag-h768-p104m-b2m-step-29999",
    },
}
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_official_train_macro_trajectory"),
    )
    return parser.parse_args()


def load_trajectory(results_root: str) -> pl.DataFrame:
    """Load the six official macro rows and assert their protocol/support parity."""
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for display_step, checkpoint in MODEL_CHECKPOINTS[model].items():
            path = f"{results_root.rstrip('/')}/{checkpoint}/mendelian_traits.parquet"
            metrics = pl.read_parquet(path)
            selected = metrics.filter(
                (pl.col("score_type") == "minus_llr_avg")
                & (pl.col("subset") == "_macro_avg_")
            )
            assert selected.height == 1, (path, selected)
            row = selected.row(0, named=True)
            rows.append(
                {
                    "model": model,
                    "step": display_step,
                    "auprc": row["value"],
                    "se": row["se"],
                    "n_groups": row["n_groups"],
                    "n_rows": row["n_rows"],
                }
            )

    trajectory = pl.DataFrame(rows).sort("model", "step")
    assert trajectory.height == 6
    assert trajectory.group_by("model").len()["len"].sort().to_list() == [3, 3]
    assert trajectory["n_groups"].n_unique() == 1
    assert trajectory["n_rows"].n_unique() == 1
    assert trajectory.filter(
        ~pl.col("auprc").is_finite()
        | ~pl.col("se").is_finite()
        | (pl.col("auprc") < 0)
        | (pl.col("auprc") > 1)
        | (pl.col("se") < 0)
    ).is_empty()
    return trajectory


def _format_step(value: float, _position: int) -> str:
    return f"{value / 1_000:g}k"


def plot_trajectory(trajectory: pl.DataFrame, output_dir: Path) -> None:
    """Render both model sizes on a shared, level-comparable AUPRC axis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    data = trajectory.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=data,
        x="step",
        y="auprc",
        hue="model",
        hue_order=MODEL_ORDER,
        kind="line",
        marker="o",
        dashes=False,
        palette=MODEL_COLORS,
        height=4.6,
        aspect=1.55,
    )
    axis = grid.ax
    for model, color in MODEL_COLORS.items():
        model_rows = data[data["model"] == model]
        assert len(model_rows) == 3
        axis.errorbar(
            model_rows["step"],
            model_rows["auprc"],
            yerr=model_rows["se"],
            fmt="none",
            capsize=0,
            color=color,
            alpha=0.7,
            linewidth=1.2,
        )
    axis.set_xticks([10_000, 20_000, 30_000])
    axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
    grid.set_axis_labels("Training step", "Macro AUPRC")
    grid.figure.suptitle("Ortholog-RAG Mendelian performance during training")
    grid.figure.text(
        0.5,
        0.01,
        "Official evals_v2 train split; zero-shot minus-LLR with forward/reverse-complement averaging; error bars = ±1 SE.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.88, bottom=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    trajectory = load_trajectory(args.results_root)
    plot_trajectory(trajectory, args.output_dir)
    print(trajectory)


if __name__ == "__main__":
    main()
