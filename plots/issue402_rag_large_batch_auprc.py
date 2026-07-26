#!/usr/bin/env python3
"""Plot issue #402 large-batch offline AUPRC by step and processed tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import polars as pl
import seaborn as sns
import wandb

INPUT_ROOTS = {
    "46M": ("gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/2026.07.26.5"),
    "104M": (
        "gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-b2m-30k/2026.07.26.5"
    ),
}
LOSS_KEY = "eval/datasets/dna-exp402-rag-tokenized/loss"
RUNS = {
    "46M": "gonzalobenegas/marin/dna-exp402-rag-h640-p46M-B2M-30K-scratch",
    "104M": "gonzalobenegas/marin/dna-exp402-rag-h768-p104M-B2M-30K-scratch",
}
EVAL_STEPS = (5_000, 10_000, 15_000, 20_000, 25_000, 29_999)
TOKENS_PER_UPDATE = 1_024 * 2_048
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}
BENCHMARK_ORDER = ["Mendelian", "Complex", "SGE"]
BENCHMARK_TITLES = {
    "Mendelian": "Mendelian (global)",
    "Complex": "Complex (global)",
    "SGE": "SGE (macro)",
}
AXIS_ORDER = [
    "Training step",
    "Processed tokens (billions)",
    "Validation loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-46m", default=INPUT_ROOTS["46M"])
    parser.add_argument("--input-104m", default=INPUT_ROOTS["104M"])
    parser.add_argument("--steps", type=int, nargs="+", default=EVAL_STEPS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_large_batch_auprc"),
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Allow live runs when all requested validation/eval steps exist.",
    )
    return parser.parse_args()


def _one_row(frame: pl.DataFrame, predicate: pl.Expr) -> dict[str, object]:
    selected = frame.filter(predicate)
    assert selected.height == 1, selected
    return selected.row(0, named=True)


def load_headline_metrics(
    input_roots: dict[str, str], steps: list[int] | tuple[int, ...]
) -> pl.DataFrame:
    """Load the frozen headline metric for each benchmark and checkpoint."""
    assert set(input_roots) == set(MODEL_ORDER)
    assert steps and len(set(steps)) == len(steps)
    assert set(steps) <= set(EVAL_STEPS)
    rows: list[dict[str, object]] = []
    sge_chances: list[float] = []
    for model in MODEL_ORDER:
        for step in sorted(steps):
            root = f"{input_roots[model].rstrip('/')}/step-{step}"
            mendelian = pl.read_parquet(f"{root}/mendelian_traits/metrics.parquet")
            complex_traits = pl.read_parquet(f"{root}/complex_traits/metrics.parquet")
            sge = pl.read_parquet(f"{root}/sge/metrics.parquet")
            mendelian_row = _one_row(
                mendelian,
                (pl.col("subset") == "_global_")
                & (pl.col("score_type") == "minus_llr_avg"),
            )
            complex_row = _one_row(
                complex_traits,
                (pl.col("subset") == "_global_")
                & (pl.col("score_type") == "abs_llr_avg"),
            )
            sge_row = _one_row(
                sge,
                (pl.col("metric") == "AUPRC")
                & (pl.col("subset") == "_macro_avg_")
                & (pl.col("accession") == "_macro_avg_")
                & (pl.col("gene") == "_macro_avg_")
                & (pl.col("score_type") == "minus_llr_avg"),
            )
            sge_cells = sge.filter(
                (pl.col("metric") == "AUPRC")
                & (pl.col("accession") != "_macro_avg_")
                & pl.col("subset").is_in(["missense_variant", "splicing"])
                & (pl.col("score_type") == "minus_llr_avg")
            )
            assert sge_cells.height == 6
            sge_chance = float(
                sge_cells.select((pl.col("n_pos") / pl.col("n")).mean()).item()
            )
            sge_chances.append(sge_chance)
            for benchmark, metric_row, chance in (
                ("Mendelian", mendelian_row, 0.1),
                ("Complex", complex_row, 0.1),
                ("SGE", sge_row, sge_chance),
            ):
                rows.append(
                    {
                        "model": model,
                        "step": step,
                        "completed_updates": step + 1,
                        "training_tokens": (step + 1) * TOKENS_PER_UPDATE,
                        "training_tokens_billions": (
                            (step + 1) * TOKENS_PER_UPDATE / 1e9
                        ),
                        "benchmark": benchmark,
                        "auprc": metric_row["value"],
                        "se": metric_row["se"],
                        "prevalence_reference": chance,
                    }
                )
    assert max(sge_chances) - min(sge_chances) < 1e-12
    metrics = pl.DataFrame(rows).sort("benchmark", "model", "step")
    assert metrics.height == len(MODEL_ORDER) * len(steps) * len(BENCHMARK_ORDER)
    assert metrics.filter(
        ~pl.col("auprc").is_finite()
        | ~pl.col("se").is_finite()
        | (pl.col("auprc") < 0)
        | (pl.col("auprc") > 1)
        | (pl.col("se") < 0)
    ).is_empty()
    assert metrics.group_by("model", "benchmark").len()["len"].unique().to_list() == [
        len(steps)
    ]
    return metrics


def load_exact_validation_losses(
    api: wandb.Api,
    steps: list[int] | tuple[int, ...],
    *,
    allow_running: bool,
) -> pl.DataFrame:
    """Load the validation loss logged at each offline-eval checkpoint."""
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        run = api.run(RUNS[model])
        if allow_running:
            assert run.state in {"running", "finished"}, (model, run.state)
        else:
            assert run.state == "finished", (model, run.state)
        by_step: dict[int, set[float]] = {}
        for row in run.scan_history(keys=["global_step", LOSS_KEY], page_size=1_000):
            if row.get("global_step") is None or row.get(LOSS_KEY) is None:
                continue
            step = int(row["global_step"])
            if step in steps:
                by_step.setdefault(step, set()).add(float(row[LOSS_KEY]))
        assert set(by_step) == set(steps), (model, sorted(by_step))
        for step in sorted(steps):
            assert len(by_step[step]) == 1, (model, step, by_step[step])
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "validation_loss": by_step[step].pop(),
                }
            )
    losses = pl.DataFrame(rows)
    assert losses.height == len(MODEL_ORDER) * len(steps)
    assert losses.filter(
        ~pl.col("validation_loss").is_finite() | (pl.col("validation_loss") <= 0)
    ).is_empty()
    return losses


def _format_step(value: float, _position: int) -> str:
    return f"{value / 1_000:g}k"


def _format_billions(value: float, _position: int) -> str:
    return f"{value:g}B"


def plot_headline_metrics(metrics: pl.DataFrame, output_dir: Path) -> None:
    """Render checkpoint AUPRC on both progress scales."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    plot_data = pl.concat(
        [
            metrics.with_columns(
                pl.col("step").cast(pl.Float64).alias("progress"),
                pl.lit(AXIS_ORDER[0]).alias("progress_axis"),
            ),
            metrics.with_columns(
                pl.col("training_tokens_billions").alias("progress"),
                pl.lit(AXIS_ORDER[1]).alias("progress_axis"),
            ),
            metrics.with_columns(
                pl.col("validation_loss").alias("progress"),
                pl.lit(AXIS_ORDER[2]).alias("progress_axis"),
            ),
        ]
    ).to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=plot_data,
        x="progress",
        y="auprc",
        hue="model",
        hue_order=MODEL_ORDER,
        style="model",
        style_order=MODEL_ORDER,
        row="progress_axis",
        row_order=AXIS_ORDER,
        col="benchmark",
        col_order=BENCHMARK_ORDER,
        kind="line",
        markers=True,
        dashes=False,
        palette=MODEL_COLORS,
        height=3.35,
        aspect=1.12,
        facet_kws={"sharex": False, "sharey": False},
    )
    grid.set_axis_labels("", "AUPRC")
    for (axis_name, benchmark), axis in grid.axes_dict.items():
        axis.set_title(
            f"{BENCHMARK_TITLES[benchmark]}\n{axis_name}",
            fontsize=11,
        )
        if axis_name == AXIS_ORDER[0]:
            axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
            x_column = "step"
        elif axis_name == AXIS_ORDER[1]:
            axis.xaxis.set_major_formatter(FuncFormatter(_format_billions))
            x_column = "training_tokens_billions"
        else:
            x_column = "validation_loss"
            axis.invert_xaxis()
        subset = metrics.filter(pl.col("benchmark") == benchmark).to_pandas()
        reference = subset["prevalence_reference"].unique()
        assert len(reference) == 1
        axis.axhline(
            float(reference[0]),
            color="#737373",
            linestyle=":",
            linewidth=1,
        )
        for model in MODEL_ORDER:
            model_rows = subset[subset["model"] == model]
            axis.errorbar(
                model_rows[x_column],
                model_rows["auprc"],
                yerr=model_rows["se"],
                fmt="none",
                capsize=0,
                color=MODEL_COLORS[model],
                alpha=0.65,
                linewidth=1.05,
            )
    grid.figure.suptitle("Large-batch ortholog-RAG offline AUPRC by training progress")
    grid.figure.text(
        0.5,
        0.01,
        "Error bars = ±1 bootstrap SE; dotted lines = fixed prevalence. "
        "Loss decreases left-to-right in the bottom row. Each benchmark/row "
        "y-axis is independently scaled. Complex uses "
        "|mean(forward LLR, reverse-complement LLR)|.",
        ha="center",
        fontsize=9.5,
    )
    grid.figure.subplots_adjust(top=0.9, bottom=0.075, hspace=0.45, wspace=0.28)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    steps = list(args.steps)
    metrics = load_headline_metrics(
        {"46M": args.input_46m, "104M": args.input_104m},
        steps,
    ).join(
        load_exact_validation_losses(
            wandb.Api(),
            steps,
            allow_running=args.allow_running,
        ),
        on=["model", "step"],
        how="inner",
        validate="m:1",
    )
    assert metrics.height == len(MODEL_ORDER) * len(steps) * len(BENCHMARK_ORDER)
    plot_headline_metrics(metrics, args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
