#!/usr/bin/env python3
"""Plot issue #402 full-context versus ortholog-free VEP performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

ROOTS = {
    "46M": ("gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-30k/2026.07.26"),
    "104M": ("gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-30k/2026.07.26"),
}
SANITY_ROOTS = {model: f"{root}/sanity-ac7016" for model, root in ROOTS.items()}
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}
BENCHMARKS = {
    "Mendelian": ("mendelian_traits", "minus_llr_avg"),
    "Complex": ("complex_traits", "abs_llr_avg"),
    "SGE": ("sge", "minus_llr_avg"),
}
CONTEXTS = [
    ("full", "Full\northologs"),
    ("all_n", "All N\n(fixed 2048)"),
    ("human_only", "Human only\n(OOD 256)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-46m", default=ROOTS["46M"])
    parser.add_argument("--eval-104m", default=ROOTS["104M"])
    parser.add_argument("--sanity-46m", default=SANITY_ROOTS["46M"])
    parser.add_argument("--sanity-104m", default=SANITY_ROOTS["104M"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_vep_context_ablation"),
    )
    return parser.parse_args()


def _metric_row(
    metrics: pl.DataFrame, benchmark: str, score_type: str
) -> dict[str, object]:
    if benchmark == "SGE":
        selected = metrics.filter(
            (pl.col("metric") == "AUPRC")
            & (pl.col("subset") == "_macro_avg_")
            & (pl.col("accession") == "_macro_avg_")
            & (pl.col("gene") == "_macro_avg_")
            & (pl.col("score_type") == score_type)
        )
    else:
        selected = metrics.filter(
            (pl.col("subset") == "_global_") & (pl.col("score_type") == score_type)
        )
    assert selected.height == 1, selected
    return {
        "auprc": float(selected["value"].item()),
        "se": float(selected["se"].item()),
    }


def _random_baseline(sanity_root: str, benchmark: str, slug: str) -> float:
    variants = pl.read_parquet(
        f"{sanity_root.rstrip('/')}/vep/{slug}/all_n/variants.parquet"
    )
    if benchmark == "SGE":
        value = (
            variants.group_by("mavedb_urn", "subset")
            .agg(pl.col("label").mean().alias("prevalence"))
            .group_by("mavedb_urn")
            .agg(pl.col("prevalence").mean().alias("prevalence"))
            .select(pl.col("prevalence").mean())
            .item()
        )
    else:
        value = variants["label"].mean()
    assert value is not None and 0 < value < 1
    return float(value)


def load_context_results(
    eval_roots: dict[str, str] = ROOTS,
    sanity_roots: dict[str, str] = SANITY_ROOTS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load corrected full-context and both ortholog-free controls."""
    assert set(eval_roots) == set(MODEL_ORDER)
    assert set(sanity_roots) == set(MODEL_ORDER)
    rows: list[dict[str, object]] = []
    for model, root in eval_roots.items():
        for benchmark, (slug, score_type) in BENCHMARKS.items():
            for context_index, (context, context_label) in enumerate(CONTEXTS):
                if context == "full":
                    path = f"{root.rstrip('/')}/step-29999/{slug}/metrics.parquet"
                else:
                    sanity_root = sanity_roots[model].rstrip("/")
                    path = f"{sanity_root}/vep/{slug}/{context}/metrics.parquet"
                rows.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "context": context,
                        "context_label": context_label,
                        "context_index": context_index,
                        **_metric_row(pl.read_parquet(path), benchmark, score_type),
                    }
                )
    data = pl.DataFrame(rows)
    assert data.height == len(eval_roots) * len(BENCHMARKS) * len(CONTEXTS)
    assert data.filter(
        ~pl.col("auprc").is_finite() | ~pl.col("se").is_finite()
    ).is_empty()
    baseline_root = sanity_roots["46M"]
    baselines = pl.DataFrame(
        [
            {
                "benchmark": benchmark,
                "random_auprc": _random_baseline(baseline_root, benchmark, slug),
            }
            for benchmark, (slug, _) in BENCHMARKS.items()
        ]
    )
    assert baselines.height == len(BENCHMARKS)
    return data, baselines


def plot_context_results(
    data: pl.DataFrame, baselines: pl.DataFrame, output_dir: Path
) -> None:
    """Render full and ortholog-free VEP performance with random baselines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    baselines.write_parquet(output_dir / "baselines.parquet", compression="zstd")
    frame = data.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="context_index",
        y="auprc",
        hue="model",
        hue_order=MODEL_ORDER,
        style="model",
        style_order=MODEL_ORDER,
        col="benchmark",
        col_order=list(BENCHMARKS),
        kind="line",
        markers=True,
        dashes=False,
        palette=MODEL_COLORS,
        height=4.4,
        aspect=1.05,
        facet_kws={"sharex": True, "sharey": False},
    )
    grid.set_axis_labels("", "AUPRC")
    context_labels = [label for _, label in CONTEXTS]
    for benchmark, axis in grid.axes_dict.items():
        subset = frame[frame["benchmark"] == benchmark]
        axis.set_title(benchmark)
        axis.set_xticks(range(len(context_labels)), context_labels)
        baseline = float(
            baselines.filter(pl.col("benchmark") == benchmark)["random_auprc"].item()
        )
        axis.axhline(
            baseline,
            color="#555555",
            linestyle=":",
            linewidth=1.4,
            label="Random prevalence",
        )
        for model in MODEL_ORDER:
            model_rows = subset[subset["model"] == model].sort_values("context_index")
            axis.errorbar(
                model_rows["context_index"],
                model_rows["auprc"],
                yerr=model_rows["se"],
                fmt="none",
                ecolor=MODEL_COLORS[model],
                elinewidth=1.2,
                capsize=0,
                alpha=0.85,
            )
    grid.figure.suptitle(
        "Variant-effect performance collapses without ortholog sequence content"
    )
    grid.figure.text(
        0.5,
        0.015,
        "Points ±1 bootstrap SE. Dotted line is the task prevalence baseline; "
        "benchmark y-axes are independently scaled.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.82, bottom=0.24, wspace=0.28)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    data, baselines = load_context_results(
        {"46M": args.eval_46m, "104M": args.eval_104m},
        {"46M": args.sanity_46m, "104M": args.sanity_104m},
    )
    plot_context_results(data, baselines, args.output_dir)
    print(data.sort("benchmark", "model", "context_index"))
    print(baselines.sort("benchmark"))


if __name__ == "__main__":
    main()
