"""Plot the corrected issue #417 Mendelian and SGE terminal results."""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import matplotlib
import pandas as pd
import polars as pl
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SUMMARY_URI = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/"
    "results/comparisons/issue417/summary.json"
)
SGE_METRICS_URI = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
    "exp417-cds-mammals-only-step-4999/sge.parquet"
)
OUTPUT_DIR = Path("plots/output/issue417_vep_results")

MODEL_ORDER = ("Mammals only", "Combined vertebrates")
MODEL_PALETTE = dict(zip(MODEL_ORDER, sns.color_palette("colorblind", 2), strict=True))
FACET_LABELS = {
    ("mendelian_traits", "missense_variant"): "Mendelian · missense",
    ("mendelian_traits", "splicing"): "Mendelian · splicing",
    ("mendelian_traits", "synonymous_variant"): "Mendelian · synonymous",
    ("sge", "missense_variant"): "SGE · missense",
    ("sge", "splicing"): "SGE · splicing",
}
FACET_ORDER = tuple(FACET_LABELS.values())


def _load_summary() -> dict[str, object]:
    with fsspec.open(SUMMARY_URI, "rt") as summary_file:
        summary = json.load(summary_file)
    assert summary["split"] == "train"
    assert summary["score_type"] == "minus_llr_avg"
    assert len(summary["results"]) == 5
    return summary


def _sge_random_baselines() -> dict[str, float]:
    """Macro-average the positive prevalence over qualifying SGE accessions."""
    metrics = pl.read_parquet(SGE_METRICS_URI)
    leaf = metrics.filter(
        (pl.col("score_type") == "minus_llr_avg")
        & pl.col("subset").is_in(["missense_variant", "splicing"])
        & (pl.col("accession") != "_macro_avg_")
        & pl.col("value").is_not_null()
    ).with_columns((pl.col("n_pos") / pl.col("n")).alias("prevalence"))
    baselines = {
        row["subset"]: row["baseline"]
        for row in leaf.group_by("subset")
        .agg(pl.col("prevalence").mean().alias("baseline"))
        .to_dicts()
    }
    assert baselines.keys() == {"missense_variant", "splicing"}
    return baselines


def _plot_frame() -> tuple[pd.DataFrame, dict[str, float]]:
    summary = _load_summary()
    sge_baselines = _sge_random_baselines()
    records: list[dict[str, object]] = []
    baselines: dict[str, float] = {}
    for row in summary["results"]:
        key = (row["dataset"], row["scope"])
        assert key in FACET_LABELS
        facet = FACET_LABELS[key]
        baseline = (
            0.1 if row["dataset"] == "mendelian_traits" else sge_baselines[row["scope"]]
        )
        baselines[facet] = baseline
        records.extend(
            [
                {
                    "facet": facet,
                    "model": "Mammals only",
                    "value": row["mammals_value"],
                    "se": row["mammals_se"],
                },
                {
                    "facet": facet,
                    "model": "Combined vertebrates",
                    "value": row["combined_value"],
                    "se": row["combined_se"],
                },
            ]
        )
    frame = pd.DataFrame.from_records(records)
    assert len(frame) == 10
    assert frame[["value", "se"]].notna().all().all()
    assert set(frame["facet"]) == set(FACET_ORDER)
    return frame, baselines


def main() -> None:
    frame, baselines = _plot_frame()
    sns.set_theme(style="ticks", context="talk")
    grid = sns.catplot(
        data=frame,
        x="model",
        y="value",
        hue="model",
        order=MODEL_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        col="facet",
        col_order=FACET_ORDER,
        col_wrap=3,
        kind="bar",
        errorbar=None,
        legend=False,
        sharey=False,
        height=3.35,
        aspect=0.95,
    )
    grid.set_axis_labels("", "AUPRC")
    grid.set_titles("")

    for facet, axis in grid.axes_dict.items():
        axis.set_title(facet, pad=12, fontsize=16)
        facet_rows = (
            frame[frame["facet"] == facet].set_index("model").loc[list(MODEL_ORDER)]
        )
        baseline = baselines[facet]
        observed_top = max((facet_rows["value"] + facet_rows["se"]).tolist())
        span = observed_top - baseline
        upper = observed_top + max(0.025, 0.22 * span)
        axis.set_ylim(baseline, upper)
        for patch, (_, row) in zip(axis.patches, facet_rows.iterrows(), strict=True):
            center = patch.get_x() + patch.get_width() / 2
            axis.errorbar(
                center,
                row["value"],
                yerr=row["se"],
                color="0.15",
                linewidth=1.3,
                capsize=0,
                zorder=4,
            )
            axis.text(
                center,
                row["value"] + row["se"] + 0.035 * (upper - baseline),
                f"{row['value']:.3f}",
                ha="center",
                va="bottom",
                fontsize="small",
            )
        axis.set_xticks([])
        sns.despine(ax=axis)

    grid.figure.suptitle(
        "Projected vertebrate CDS models at step 4,999",
        y=0.98,
    )
    grid.figure.legend(
        handles=[
            Patch(facecolor=MODEL_PALETTE[model], label=model) for model in MODEL_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    grid.figure.text(
        0.5,
        0.005,
        "Official evals_v2 train split · error bars = ±1 SE (paired bootstrap units)",
        ha="center",
        fontsize="small",
    )
    grid.figure.subplots_adjust(bottom=0.14, top=0.76, wspace=0.30, hspace=0.72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(OUTPUT_DIR / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(OUTPUT_DIR / "figure.png", bbox_inches="tight", dpi=180)
    plt.close(grid.figure)


if __name__ == "__main__":
    main()
