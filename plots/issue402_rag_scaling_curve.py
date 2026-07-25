#!/usr/bin/env python3
"""Compare issue #402's 46M and 104M offline AUPRC checkpoint curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import polars as pl
import seaborn as sns

DEFAULT_INPUT_ROOTS = {
    "46M": (
        "gs://marin-us-east5/users/ubuntu/evals/dna-exp402-rag-h640-p46m-1b/ropefix"
    ),
    "104M": (
        "gs://marin-us-east5/users/ubuntu/evals/dna-exp402-rag-h768-p104m-1b/ropefix"
    ),
}
DEFAULT_STEPS = (1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 7_628)
BENCHMARK_TITLES = {
    "Mendelian": "Mendelian traits",
    "Complex": "Complex traits",
    "SGE": "SGE",
}
AGGREGATE_COLORS = {
    "global": "#3366cc",
    "macro": "#d95f02",
}
MODEL_STYLES = {
    "46M": {"linestyle": "-", "marker": "o"},
    "104M": {"linestyle": "--", "marker": "s"},
}


def _format_training_step(value: float, _position: int) -> str:
    """Render dense checkpoint ticks without overlapping four-digit labels."""
    return f"{value / 1_000:g}k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-46m", default=DEFAULT_INPUT_ROOTS["46M"])
    parser.add_argument("--input-104m", default=DEFAULT_INPUT_ROOTS["104M"])
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_scaling_curve"),
    )
    return parser.parse_args()


def _one_row(df: pl.DataFrame, predicate: pl.Expr) -> dict[str, object]:
    selected = df.filter(predicate)
    assert selected.height == 1, selected
    return selected.row(0, named=True)


def load_curves(
    input_roots: dict[str, str], steps: list[int] | tuple[int, ...]
) -> pl.DataFrame:
    """Load both size rungs and assert identical metric contracts."""
    assert set(input_roots) == set(MODEL_STYLES)
    assert steps
    assert len(set(steps)) == len(steps)
    rows: list[dict[str, object]] = []
    sge_chance_values: list[float] = []
    for model_size, input_root in input_roots.items():
        for step in sorted(steps):
            step_root = f"{input_root.rstrip('/')}/step-{step}"
            for benchmark, directory in (
                ("Mendelian", "mendelian_traits"),
                ("Complex", "complex_traits"),
            ):
                metrics = pl.read_parquet(f"{step_root}/{directory}/metrics.parquet")
                expected_score = (
                    "minus_llr_avg" if benchmark == "Mendelian" else "abs_llr_avg"
                )
                assert metrics["score_type"].unique().to_list() == [expected_score]
                for subset, aggregate in (
                    ("_global_", "global"),
                    ("_macro_avg_", "macro"),
                ):
                    row = _one_row(metrics, pl.col("subset") == subset)
                    rows.append(
                        {
                            "model_size": model_size,
                            "step": step,
                            "benchmark": benchmark,
                            "aggregate": aggregate,
                            "value": row["value"],
                            "se": row["se"],
                            "prevalence_reference": 0.1,
                        }
                    )

            sge = pl.read_parquet(f"{step_root}/sge/metrics.parquet")
            assert sge["score_type"].unique().to_list() == ["minus_llr_avg"]
            sge_macro = _one_row(
                sge,
                (pl.col("metric") == "AUPRC")
                & (pl.col("subset") == "_macro_avg_")
                & (pl.col("accession") == "_macro_avg_")
                & (pl.col("gene") == "_macro_avg_"),
            )
            sge_cells = sge.filter(
                (pl.col("metric") == "AUPRC")
                & (pl.col("accession") != "_macro_avg_")
                & pl.col("subset").is_in(["missense_variant", "splicing"])
            )
            assert sge_cells.height == 6
            sge_chance = sge_cells.select((pl.col("n_pos") / pl.col("n")).mean()).item()
            sge_chance_values.append(float(sge_chance))
            rows.append(
                {
                    "model_size": model_size,
                    "step": step,
                    "benchmark": "SGE",
                    "aggregate": "macro",
                    "value": sge_macro["value"],
                    "se": sge_macro["se"],
                    "prevalence_reference": sge_chance,
                }
            )

    assert max(sge_chance_values) - min(sge_chance_values) < 1e-12
    curves = pl.DataFrame(rows).sort("benchmark", "aggregate", "model_size", "step")
    assert curves.filter(
        ~pl.col("value").is_finite() | ~pl.col("se").is_finite()
    ).is_empty()
    expected_rows_per_model = len(steps) * 5
    assert curves.group_by("model_size").len()["len"].to_list() == [
        expected_rows_per_model,
        expected_rows_per_model,
    ]
    return curves


def _plot_series(
    axis: plt.Axes,
    rows: object,
    *,
    color: str,
    model_size: str,
) -> None:
    style = MODEL_STYLES[model_size]
    axis.plot(
        rows["step"],
        rows["value"],
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
    )
    axis.errorbar(
        rows["step"],
        rows["value"],
        yerr=rows["se"],
        fmt="none",
        capsize=0,
        color=color,
        alpha=0.65,
        linewidth=1.1,
    )


def plot_curves(curves: pl.DataFrame, output_dir: Path) -> None:
    """Render model-size styles with independent global/macro y-axes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    data = curves.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(3, 1, figsize=(12.0, 10.3), sharex=True)
    twin_axes: list[plt.Axes] = []
    for benchmark, axis in zip(("Mendelian", "Complex", "SGE"), axes, strict=True):
        subset = data[data["benchmark"] == benchmark]
        primary_aggregate = "macro" if benchmark == "SGE" else "global"
        primary_color = AGGREGATE_COLORS[primary_aggregate]
        for model_size in MODEL_STYLES:
            rows = subset[
                (subset["aggregate"] == primary_aggregate)
                & (subset["model_size"] == model_size)
            ]
            assert len(rows) == subset["step"].nunique()
            _plot_series(
                axis,
                rows,
                color=primary_color,
                model_size=model_size,
            )
        reference_values = subset[subset["aggregate"] == primary_aggregate][
            "prevalence_reference"
        ].unique()
        assert len(reference_values) == 1
        axis.axhline(
            float(reference_values[0]),
            color=primary_color,
            linestyle=":",
            linewidth=1,
        )
        axis.set_ylabel(
            f"{primary_aggregate.title()} AUPRC",
            color=primary_color,
        )
        axis.tick_params(axis="y", colors=primary_color)
        axis.spines["left"].set_color(primary_color)
        axis.set_title(BENCHMARK_TITLES[benchmark])
        axis.xaxis.set_major_formatter(FuncFormatter(_format_training_step))

        if benchmark != "SGE":
            macro_axis = axis.twinx()
            for model_size in MODEL_STYLES:
                rows = subset[
                    (subset["aggregate"] == "macro")
                    & (subset["model_size"] == model_size)
                ]
                assert len(rows) == subset["step"].nunique()
                _plot_series(
                    macro_axis,
                    rows,
                    color=AGGREGATE_COLORS["macro"],
                    model_size=model_size,
                )
            macro_reference_values = subset[subset["aggregate"] == "macro"][
                "prevalence_reference"
            ].unique()
            assert len(macro_reference_values) == 1
            macro_axis.axhline(
                float(macro_reference_values[0]),
                color=AGGREGATE_COLORS["macro"],
                linestyle=":",
                linewidth=1,
            )
            macro_axis.set_ylabel("Macro AUPRC", color=AGGREGATE_COLORS["macro"])
            macro_axis.tick_params(axis="y", colors=AGGREGATE_COLORS["macro"])
            macro_axis.spines["right"].set_color(AGGREGATE_COLORS["macro"])
            macro_axis.grid(False)
            twin_axes.append(macro_axis)

    assert len(twin_axes) == 2
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=model_size,
        )
        for model_size, style in MODEL_STYLES.items()
    ]
    figure.legend(
        handles=legend_handles,
        title="Parameters",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Ortholog-RAG offline variant metrics by model size",
        y=0.995,
    )
    figure.supxlabel("Training step", y=0.055)
    figure.text(
        0.5,
        0.008,
        "Error bars = ±1 SE; dotted lines = fixed prevalence on the corresponding "
        "axis. Global/macro y-axes are independent.",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(top=0.78, bottom=0.14, hspace=0.45, right=0.84)
    figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    curves = load_curves({"46M": args.input_46m, "104M": args.input_104m}, args.steps)
    plot_curves(curves, args.output_dir)
    print(curves)


if __name__ == "__main__":
    main()
