#!/usr/bin/env python3
"""Plot issue #402 checkpoint AUPRC against exact-step W&B validation loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import polars as pl
import seaborn as sns
import wandb

STEPS = (1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 7_628)
LOSS_KEY = "eval/datasets/dna-exp402-rag-tokenized/loss"
MODELS = {
    "46M": {
        "wandb": "gonzalobenegas/marin/dna-exp402-rag-h640-p46M-1B",
        "metrics": (
            "gs://marin-us-east5/users/ubuntu/evals/dna-exp402-rag-h640-p46m-1b/ropefix"
        ),
    },
    "104M": {
        "wandb": "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-1B",
        "metrics": (
            "gs://marin-us-east5/users/ubuntu/evals/"
            "dna-exp402-rag-h768-p104m-1b/ropefix"
        ),
    },
}
BENCHMARK_TITLES = {
    "Mendelian": "Mendelian (global)",
    "Complex": "Complex (global)",
    "SGE": "SGE (macro)",
}
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_loss_auprc"),
    )
    return parser.parse_args()


def _one_value(frame: pl.DataFrame, predicate: pl.Expr) -> float:
    selected = frame.filter(predicate)
    assert selected.height == 1, selected
    value = float(selected["value"].item())
    assert value == value
    return value


def load_validation_losses(api: wandb.Api) -> pl.DataFrame:
    """Fetch the pinned run histories and retain the eight VEP checkpoint steps."""
    rows: list[dict[str, object]] = []
    for model, sources in MODELS.items():
        run = api.run(str(sources["wandb"]))
        by_step: dict[int, set[float]] = {}
        for row in run.scan_history(keys=["global_step", LOSS_KEY], page_size=1_000):
            if row.get("global_step") is None or row.get(LOSS_KEY) is None:
                continue
            step = int(row["global_step"])
            if step in STEPS:
                by_step.setdefault(step, set()).add(float(row[LOSS_KEY]))
        assert set(by_step) == set(STEPS), (model, sorted(by_step))
        for step in STEPS:
            assert len(by_step[step]) == 1, (model, step, by_step[step])
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "validation_loss": by_step[step].pop(),
                }
            )
    result = pl.DataFrame(rows)
    assert result.height == len(MODELS) * len(STEPS)
    return result


def load_auprc() -> pl.DataFrame:
    """Load the frozen corrected likelihood metrics at matching checkpoints."""
    rows: list[dict[str, object]] = []
    for model, sources in MODELS.items():
        metrics_root = str(sources["metrics"])
        for step in STEPS:
            root = f"{metrics_root}/step-{step}"
            mendelian = pl.read_parquet(f"{root}/mendelian_traits/metrics.parquet")
            complex_traits = pl.read_parquet(f"{root}/complex_traits/metrics.parquet")
            sge = pl.read_parquet(f"{root}/sge/metrics.parquet")
            values = {
                "Mendelian": _one_value(
                    mendelian,
                    (pl.col("subset") == "_global_")
                    & (pl.col("score_type") == "minus_llr_avg"),
                ),
                "Complex": _one_value(
                    complex_traits,
                    (pl.col("subset") == "_global_")
                    & (pl.col("score_type") == "abs_llr_avg"),
                ),
                "SGE": _one_value(
                    sge,
                    (pl.col("metric") == "AUPRC")
                    & (pl.col("subset") == "_macro_avg_")
                    & (pl.col("accession") == "_macro_avg_")
                    & (pl.col("gene") == "_macro_avg_")
                    & (pl.col("score_type") == "minus_llr_avg"),
                ),
            }
            rows.extend(
                {
                    "model": model,
                    "step": step,
                    "benchmark": benchmark,
                    "auprc": value,
                }
                for benchmark, value in values.items()
            )
    result = pl.DataFrame(rows)
    assert result.height == len(MODELS) * len(STEPS) * len(BENCHMARK_TITLES)
    return result


def load_joined() -> pl.DataFrame:
    losses = load_validation_losses(wandb.Api())
    auprc = load_auprc()
    joined = auprc.join(losses, on=["model", "step"], how="inner", validate="m:1")
    assert joined.height == auprc.height
    assert joined.filter(
        ~pl.col("auprc").is_finite() | ~pl.col("validation_loss").is_finite()
    ).is_empty()
    return joined.sort("benchmark", "model", "step")


def _format_step(step: int) -> str:
    return f"{step / 1_000:.3g}k"


def _format_loss(value: float, _position: int) -> str:
    return f"{value:.3f}"


def plot_joined(joined: pl.DataFrame, output_dir: Path) -> None:
    """Render one independently scaled AUPRC-versus-loss facet per benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    data = joined.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=data,
        x="validation_loss",
        y="auprc",
        hue="model",
        hue_order=["46M", "104M"],
        style="model",
        style_order=["46M", "104M"],
        col="benchmark",
        col_order=["Mendelian", "Complex", "SGE"],
        kind="line",
        markers=True,
        dashes=False,
        palette=MODEL_COLORS,
        facet_kws={"sharex": True, "sharey": False},
        height=4.2,
        aspect=1.05,
    )
    grid.set_axis_labels("", "AUPRC")
    for benchmark, axis in grid.axes_dict.items():
        subset = data[data["benchmark"] == benchmark]
        assert len(subset) == len(MODELS) * len(STEPS)
        axis.set_title(BENCHMARK_TITLES[benchmark])
        axis.invert_xaxis()
        axis.xaxis.set_major_formatter(FuncFormatter(_format_loss))
        for _, row in subset.iterrows():
            if int(row["step"]) not in {1_000, 3_000, 5_000, 7_628}:
                continue
            axis.annotate(
                _format_step(int(row["step"])),
                (row["validation_loss"], row["auprc"]),
                xytext=(3, 5 if row["model"] == "46M" else -12),
                textcoords="offset points",
                fontsize=7,
                alpha=0.75,
            )
    grid.figure.suptitle("Ortholog-RAG AUPRC versus exact-step validation loss")
    grid.figure.supxlabel("Validation loss", y=0.075)
    grid.figure.text(
        0.5,
        0.015,
        "Loss decreases left-to-right (x-axis reversed); labels show training step. "
        "Each benchmark y-axis is independently scaled.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.84, bottom=0.22, wspace=0.28)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    joined = load_joined()
    plot_joined(joined, args.output_dir)
    print(joined)


if __name__ == "__main__":
    main()
