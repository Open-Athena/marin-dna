#!/usr/bin/env python3
"""Plot issue #402's 46M offline AUPRC checkpoint curves from GCS Parquets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import polars as pl
import seaborn as sns

DEFAULT_INPUT_ROOT = (
    "gs://marin-us-east5/users/ubuntu/evals/dna-exp402-rag-h640-p46m-1b/ropefix"
)
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


def _format_training_step(value: float, _position: int) -> str:
    """Render dense checkpoint ticks without overlapping four-digit labels."""
    return f"{value / 1_000:g}k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_checkpoint_curve"),
    )
    return parser.parse_args()


def _one_row(df: pl.DataFrame, predicate: pl.Expr) -> dict[str, object]:
    selected = df.filter(predicate)
    assert selected.height == 1, selected
    return selected.row(0, named=True)


def load_curve(input_root: str, steps: list[int] | tuple[int, ...]) -> pl.DataFrame:
    """Load frozen aggregate rows and assert metric/schema invariants."""
    assert steps
    assert len(set(steps)) == len(steps)
    rows: list[dict[str, object]] = []
    sge_chance_values: list[float] = []
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
                "step": step,
                "benchmark": "SGE",
                "aggregate": "macro",
                "value": sge_macro["value"],
                "se": sge_macro["se"],
                "prevalence_reference": sge_chance,
            }
        )

    assert max(sge_chance_values) - min(sge_chance_values) < 1e-12
    curve = pl.DataFrame(rows).sort("benchmark", "aggregate", "step")
    assert curve.filter(
        ~pl.col("value").is_finite() | ~pl.col("se").is_finite()
    ).is_empty()
    return curve


def plot_curve(curve: pl.DataFrame, output_dir: Path) -> None:
    """Render benchmark facets with independent global/macro y-axes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    curve.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    data = curve.to_pandas()
    primary = data[
        ((data["benchmark"] != "SGE") & (data["aggregate"] == "global"))
        | ((data["benchmark"] == "SGE") & (data["aggregate"] == "macro"))
    ]
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=primary,
        x="step",
        y="value",
        hue="aggregate",
        hue_order=["global", "macro"],
        row="benchmark",
        row_order=["Mendelian", "Complex", "SGE"],
        kind="line",
        marker="o",
        dashes=False,
        palette=AGGREGATE_COLORS,
        facet_kws={"sharex": True, "sharey": False},
        height=2.8,
        aspect=2.4,
    )
    grid.set_axis_labels("", "AUPRC")
    twin_axes = []
    for benchmark, axis in grid.axes_dict.items():
        subset = data[data["benchmark"] == benchmark]
        primary_aggregate = "macro" if benchmark == "SGE" else "global"
        primary_rows = subset[subset["aggregate"] == primary_aggregate]
        assert len(primary_rows) == subset["step"].nunique()
        primary_color = AGGREGATE_COLORS[primary_aggregate]
        axis.errorbar(
            primary_rows["step"],
            primary_rows["value"],
            yerr=primary_rows["se"],
            fmt="none",
            capsize=0,
            color=primary_color,
            alpha=0.7,
            linewidth=1.2,
        )
        reference = float(primary_rows["prevalence_reference"].iloc[0])
        assert (primary_rows["prevalence_reference"] == reference).all()
        axis.axhline(reference, color=primary_color, linestyle="--", linewidth=1)
        axis.set_ylabel(
            f"{primary_aggregate.title()} AUPRC",
            color=primary_color,
        )
        axis.tick_params(axis="y", colors=primary_color)
        axis.spines["left"].set_color(primary_color)
        axis.set_title(BENCHMARK_TITLES[benchmark])
        axis.tick_params(axis="x", labelrotation=0)
        axis.xaxis.set_major_formatter(FuncFormatter(_format_training_step))

        if benchmark != "SGE":
            macro_rows = subset[subset["aggregate"] == "macro"]
            assert len(macro_rows) == len(primary_rows)
            macro_axis = axis.twinx()
            macro_axis.plot(
                macro_rows["step"],
                macro_rows["value"],
                color=AGGREGATE_COLORS["macro"],
                marker="o",
            )
            macro_axis.errorbar(
                macro_rows["step"],
                macro_rows["value"],
                yerr=macro_rows["se"],
                fmt="none",
                capsize=0,
                color=AGGREGATE_COLORS["macro"],
                alpha=0.7,
                linewidth=1.2,
            )
            macro_reference = float(macro_rows["prevalence_reference"].iloc[0])
            assert (macro_rows["prevalence_reference"] == macro_reference).all()
            macro_axis.axhline(
                macro_reference,
                color=AGGREGATE_COLORS["macro"],
                linestyle="--",
                linewidth=1,
            )
            macro_axis.set_ylabel("Macro AUPRC", color=AGGREGATE_COLORS["macro"])
            macro_axis.tick_params(axis="y", colors=AGGREGATE_COLORS["macro"])
            macro_axis.spines["right"].set_color(AGGREGATE_COLORS["macro"])
            macro_axis.grid(False)
            twin_axes.append(macro_axis)
    assert len(twin_axes) == 2
    assert grid._legend is not None
    grid._legend.remove()
    grid.figure.suptitle("46M ortholog-RAG offline variant metrics across training")
    grid.figure.supxlabel("Training step", y=0.065)
    grid.figure.text(
        0.5,
        0.008,
        "Error bars = ±1 SE; dashed lines = fixed prevalence on the corresponding "
        "axis. Global/macro y-axes are independent.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.91, bottom=0.16, hspace=0.45)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    curve = load_curve(args.input_root, args.steps)
    plot_curve(curve, args.output_dir)
    print(curve)


if __name__ == "__main__":
    main()
