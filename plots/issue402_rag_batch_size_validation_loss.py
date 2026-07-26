#!/usr/bin/env python3
"""Compare issue #402 validation loss across optimizer batch sizes."""

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
    ("46M", "131k tokens/update"): (
        "gonzalobenegas/marin/dna-exp402-rag-h640-p46M-30K-scratch",
        64 * 2_048,
    ),
    ("46M", "2.10M tokens/update"): (
        "gonzalobenegas/marin/dna-exp402-rag-h640-p46M-B2M-30K-scratch",
        1_024 * 2_048,
    ),
    ("104M", "131k tokens/update"): (
        "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-30K-scratch",
        64 * 2_048,
    ),
    ("104M", "2.10M tokens/update"): (
        "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-B2M-30K-scratch",
        1_024 * 2_048,
    ),
}
MODEL_ORDER = ["46M", "104M"]
BATCH_ORDER = ["131k tokens/update", "2.10M tokens/update"]
BATCH_COLORS = {
    "131k tokens/update": "#7f7f7f",
    "2.10M tokens/update": "#3366cc",
}
EXPECTED_STEPS = (*range(1_000, 30_000, 1_000), 29_999)
AXIS_ORDER = ["Optimizer updates", "Processed tokens (billions)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_batch_size_validation_loss"),
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Render available large-batch points instead of requiring completion.",
    )
    return parser.parse_args()


def load_validation_losses(
    api: wandb.Api, *, allow_running: bool = False
) -> pl.DataFrame:
    """Load exact scheduled validation points from both batch-size regimes."""
    rows: list[dict[str, object]] = []
    for (model, batch), (run_path, tokens_per_update) in RUNS.items():
        run = api.run(run_path)
        expected_states = {"finished", "running"} if allow_running else {"finished"}
        assert run.state in expected_states, (model, batch, run.state)
        for row in run.scan_history(keys=["global_step", LOSS_KEY], page_size=1_000):
            if row.get("global_step") is None or row.get(LOSS_KEY) is None:
                continue
            step = int(row["global_step"])
            assert step in EXPECTED_STEPS, (model, batch, step)
            rows.append(
                {
                    "model": model,
                    "batch": batch,
                    "step": step,
                    "completed_updates": step + 1,
                    "training_tokens": (step + 1) * tokens_per_update,
                    "training_tokens_billions": ((step + 1) * tokens_per_update / 1e9),
                    "validation_loss": float(row[LOSS_KEY]),
                    "run_state": run.state,
                }
            )
    data = pl.DataFrame(rows).unique(
        ["model", "batch", "step", "validation_loss", "run_state"]
    )
    assert data.filter(
        ~pl.col("validation_loss").is_finite() | (pl.col("validation_loss") <= 0)
    ).is_empty()
    assert (
        data.group_by("model", "batch", "step")
        .len()
        .filter(pl.col("len") != 1)
        .is_empty()
    )
    for model in MODEL_ORDER:
        for batch in BATCH_ORDER:
            subset = data.filter(
                (pl.col("model") == model) & (pl.col("batch") == batch)
            ).sort("step")
            assert not subset.is_empty(), (model, batch)
            assert subset["step"].to_list() == sorted(set(subset["step"]))
            if batch == BATCH_ORDER[0] or not allow_running:
                assert subset["step"].to_list() == list(EXPECTED_STEPS), (
                    model,
                    batch,
                )
    return data.sort("model", "batch", "step")


def match_large_to_small_batch(losses: pl.DataFrame) -> pl.DataFrame:
    """Match every available large-batch validation point by processed tokens."""
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        small = losses.filter(
            (pl.col("model") == model) & (pl.col("batch") == BATCH_ORDER[0])
        )
        large = losses.filter(
            (pl.col("model") == model) & (pl.col("batch") == BATCH_ORDER[1])
        )
        assert small.height == len(EXPECTED_STEPS)
        for large_row in large.iter_rows(named=True):
            matched = (
                small.with_columns(
                    (pl.col("training_tokens") - int(large_row["training_tokens"]))
                    .abs()
                    .alias("token_gap")
                )
                .sort("token_gap", "step")
                .row(0, named=True)
            )
            rows.append(
                {
                    "model": model,
                    "large_batch_step": large_row["step"],
                    "small_batch_step": matched["step"],
                    "large_batch_tokens": large_row["training_tokens"],
                    "small_batch_tokens": matched["training_tokens"],
                    "token_gap": matched["token_gap"],
                    "large_batch_loss": large_row["validation_loss"],
                    "small_batch_loss": matched["validation_loss"],
                    "large_minus_small_loss": (
                        float(large_row["validation_loss"])
                        - float(matched["validation_loss"])
                    ),
                }
            )
    matched = pl.DataFrame(rows).sort("model", "large_batch_step")
    assert matched.height == losses.filter(pl.col("batch") == BATCH_ORDER[1]).height
    assert matched.filter(
        ~pl.col("large_minus_small_loss").is_finite() | (pl.col("token_gap") < 0)
    ).is_empty()
    return matched


def _format_step(value: float, _position: int) -> str:
    return f"{value / 1_000:g}k"


def _format_billions(value: float, _position: int) -> str:
    return f"{value:g}B"


def plot_validation_losses(
    losses: pl.DataFrame, matched: pl.DataFrame, output_dir: Path
) -> None:
    """Render update- and token-domain loss curves for both model sizes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    losses.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    matched.write_parquet(output_dir / "token_matched.parquet", compression="zstd")
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
        hue="batch",
        hue_order=BATCH_ORDER,
        style="batch",
        style_order=BATCH_ORDER,
        row="model",
        row_order=MODEL_ORDER,
        col="progress_axis",
        col_order=AXIS_ORDER,
        kind="line",
        markers=True,
        dashes=False,
        palette=BATCH_COLORS,
        height=3.8,
        aspect=1.35,
        facet_kws={"sharex": False, "sharey": True},
    )
    grid.set_axis_labels("", "Validation loss")
    grid.set_titles(template="{row_name} · {col_name}")
    for (model, axis_name), axis in grid.axes_dict.items():
        if axis_name == AXIS_ORDER[0]:
            axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
        else:
            axis.xaxis.set_major_formatter(FuncFormatter(_format_billions))
        for batch in BATCH_ORDER:
            latest = losses.filter(
                (pl.col("model") == model) & (pl.col("batch") == batch)
            ).tail(1)
            x_value = (
                float(latest["step"].item())
                if axis_name == AXIS_ORDER[0]
                else float(latest["training_tokens_billions"].item())
            )
            axis.annotate(
                f"{latest['validation_loss'].item():.4f}",
                (x_value, latest["validation_loss"].item()),
                xytext=(-6, 7 if batch == BATCH_ORDER[0] else -13),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=BATCH_COLORS[batch],
            )
    qualifier = (
        "complete"
        if set(losses.filter(pl.col("batch") == BATCH_ORDER[1])["run_state"])
        == {"finished"}
        else "live partial"
    )
    grid.figure.suptitle(f"Optimizer-batch validation loss comparison ({qualifier})")
    grid.figure.text(
        0.5,
        0.008,
        "Same frozen validation set. Token-matched comparisons use the nearest "
        "scheduled small-batch checkpoint; checkpoint indices are zero-based.",
        ha="center",
        fontsize=9,
    )
    grid.figure.subplots_adjust(top=0.89, bottom=0.12, hspace=0.25, wspace=0.14)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    losses = load_validation_losses(wandb.Api(), allow_running=args.allow_running)
    matched = match_large_to_small_batch(losses)
    plot_validation_losses(losses, matched, args.output_dir)
    print(matched)


if __name__ == "__main__":
    main()
