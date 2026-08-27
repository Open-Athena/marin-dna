"""Plot the issue #517 terminal Mendelian specialist diagonal.

The input Parquets are the development-split ``evals_v2`` metric artifacts for
the six terminal step-4999 models.

Run from the repository root with an evals_v2-compatible environment:

    python .agents/artifacts/issue-517/evaluation/plot_issue517_diagonal.py

Pass ``--metrics-dir`` to reuse local copies named
``<arm>-mendelian_traits.parquet`` instead of reading workflow-owned S3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


S3_PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCORE_TYPE = "minus_llr_avg"
AUPRC_BASELINE = 0.10

ARMS: list[tuple[str, str]] = [
    ("CDS", "cds"),
    ("3′ UTR", "utr3"),
    ("ncRNA exon", "ncrna-exon"),
    ("TSS / 5′ UTR", "tss-utr5"),
    ("Enhancer A", "enhancer-arm-a"),
    ("Background", "background"),
]
SUBSETS: list[tuple[str, str]] = [
    ("Missense", "missense_variant"),
    ("Synonymous", "synonymous_variant"),
    ("Splicing", "splicing"),
    ("3′ UTR", "3_prime_UTR_variant"),
    ("ncRNA", "non_coding_transcript_exon_variant"),
    ("5′ UTR", "5_prime_UTR_variant"),
    ("Promoter", "tss_proximal"),
    ("Distal", "distal"),
]
DIAGONAL: dict[str, set[str]] = {
    "CDS": {"Missense", "Synonymous", "Splicing"},
    "3′ UTR": {"3′ UTR"},
    "ncRNA exon": {"ncRNA"},
    "TSS / 5′ UTR": {"5′ UTR", "Promoter"},
    "Enhancer A": {"Distal"},
    "Background": set(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        help="Optional directory containing locally cached metric Parquets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for the SVG and source-data CSV.",
    )
    return parser.parse_args()


def metric_path(arm_slug: str, metrics_dir: Path | None) -> str | Path:
    if metrics_dir is not None:
        return metrics_dir / f"{arm_slug}-mendelian_traits.parquet"
    model = f"exp517-gpn-uniform-{arm_slug}-step-4999"
    return f"{S3_PREFIX}/{model}/mendelian_traits.parquet"


def load_metrics(metrics_dir: Path | None) -> pd.DataFrame:
    records: list[dict[str, str | float | int]] = []
    for arm_label, arm_slug in ARMS:
        frame = pd.read_parquet(metric_path(arm_slug, metrics_dir))
        frame = frame[frame["score_type"].eq(SCORE_TYPE)].set_index("subset")
        for subset_label, subset_slug in SUBSETS:
            row = frame.loc[subset_slug]
            records.append(
                {
                    "arm": arm_label,
                    "subset": subset_label,
                    "value": float(row["value"]),
                    "se": float(row["se"]),
                    "n_groups": int(row["n_groups"]),
                    "n_rows": int(row["n_rows"]),
                }
            )
    result = pd.DataFrame.from_records(records)
    assert len(result) == len(ARMS) * len(SUBSETS)
    assert result[["value", "se"]].notna().all().all()
    assert result["value"].between(0, 1).all()
    assert (result["se"] >= 0).all()
    return result


def plot_diagonal(metrics: pd.DataFrame, output_path: Path) -> None:
    arm_order = [label for label, _ in ARMS]
    subset_order = [label for label, _ in SUBSETS]
    values = metrics.pivot(index="subset", columns="arm", values="value").loc[
        subset_order, arm_order
    ]
    errors = metrics.pivot(index="subset", columns="arm", values="se").loc[
        subset_order, arm_order
    ]
    annotations = np.asarray(
        [
            [f"{values.loc[s, a]:.3f}\n±{errors.loc[s, a]:.3f}" for a in arm_order]
            for s in subset_order
        ]
    )

    sns.set_theme(style="white", context="notebook")
    mpl.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(9.4, 8.2))
    heatmap = sns.heatmap(
        values,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="Reds",
        vmin=AUPRC_BASELINE,
        vmax=0.41,
        square=True,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "AUPRC", "shrink": 0.78},
        annot_kws={"fontsize": 8.3},
    )

    for text_artist, value in zip(heatmap.texts, values.to_numpy().ravel()):
        text_artist.set_color("white" if value >= 0.27 else "#202020")

    for row_idx, subset in enumerate(subset_order):
        for col_idx, arm in enumerate(arm_order):
            if subset in DIAGONAL[arm]:
                ax.add_patch(
                    Rectangle(
                        (col_idx, row_idx),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.2,
                    )
                )
        winner_idx = int(np.argmax(values.loc[subset].to_numpy()))
        ax.scatter(
            winner_idx + 0.79,
            row_idx + 0.21,
            marker="*",
            s=100,
            color="#7CFC00",
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )

    ax.set_title("Issue 517 terminal Mendelian AUPRC diagonal", pad=12)
    ax.set_xlabel("Training arm")
    ax.set_ylabel("Mendelian variant subset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    legend_handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2.2),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#7CFC00",
            markeredgecolor="black",
            markersize=11,
        ),
    ]
    ax.legend(
        legend_handles,
        ["Matched training scope", "Row maximum"],
        title="Markers",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(output_path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.metrics_dir)
    metrics.to_csv(args.output_dir / "issue517_mendelian_diagonal_data.csv", index=False)
    plot_diagonal(metrics, args.output_dir / "issue517_mendelian_diagonal.svg")


if __name__ == "__main__":
    main()
