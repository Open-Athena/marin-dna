#!/usr/bin/env python3
"""Plot issue #402's 46M offline AUPRC checkpoint curves from GCS Parquets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

DEFAULT_INPUT_ROOT = (
    "gs://marin-us-east5/users/ubuntu/evals/"
    "dna-exp402-rag-h640-p46m-1b/noeval2-dev"
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
        sge_chance = sge_cells.select(
            (pl.col("n_pos") / pl.col("n")).mean()
        ).item()
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
    """Render independent benchmark facets with capless ±1 SE bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    curve.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    data = curve.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=data,
        x="step",
        y="value",
        hue="aggregate",
        col="benchmark",
        col_order=["Mendelian", "Complex", "SGE"],
        kind="line",
        marker="o",
        dashes=False,
        palette=AGGREGATE_COLORS,
        facet_kws={"sharey": False},
        height=4.6,
        aspect=0.9,
    )
    grid.set_axis_labels("", "AUPRC")
    for benchmark, axis in grid.axes_dict.items():
        subset = data[data["benchmark"] == benchmark]
        for aggregate, group in subset.groupby("aggregate", sort=False):
            axis.errorbar(
                group["step"],
                group["value"],
                yerr=group["se"],
                fmt="none",
                capsize=0,
                color=AGGREGATE_COLORS[aggregate],
                alpha=0.7,
                linewidth=1.2,
            )
        reference = float(subset["prevalence_reference"].iloc[0])
        assert (subset["prevalence_reference"] == reference).all()
        axis.axhline(reference, color="#666666", linestyle="--", linewidth=1)
        axis.set_title(BENCHMARK_TITLES[benchmark])
        axis.tick_params(axis="x", labelrotation=30)
    grid.figure.suptitle("46M ortholog-RAG offline variant metrics across training")
    grid.figure.supxlabel("Training step", y=0.08)
    grid.figure.text(
        0.5,
        0.01,
        "Error bars = ±1 SE; dashed lines = fixed prevalence reference. "
        "Facet y-axes are independent.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.78, bottom=0.35)
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
