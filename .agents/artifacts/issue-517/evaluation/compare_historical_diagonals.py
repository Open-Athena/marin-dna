"""Compare issue #517 Mendelian specialist matrices with historical diagonals."""

from __future__ import annotations

from dataclasses import dataclass
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
OUTPUT_DIR = Path(".agents/artifacts/issue-517/evaluation")

ARMS: list[tuple[str, str]] = [
    ("CDS", "cds"),
    ("3′ UTR", "utr3"),
    ("ncRNA exon", "ncrna"),
    ("TSS / 5′ UTR", "tss"),
    ("Enhancer", "enhancer"),
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
HOME_ARM: dict[str, str] = {
    "Missense": "CDS",
    "Synonymous": "CDS",
    "Splicing": "CDS",
    "3′ UTR": "3′ UTR",
    "ncRNA": "ncRNA exon",
    "5′ UTR": "TSS / 5′ UTR",
    "Promoter": "TSS / 5′ UTR",
    "Distal": "Enhancer",
}


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    models: dict[str, str]


EXPERIMENTS = [
    Experiment(
        key="exp232",
        title="#232 uniform grid\nphyloP, full-window projection",
        models={
            "CDS": "exp232-v4_cds-step-4999",
            "3′ UTR": "exp232-v4_utr3-step-4999",
            "ncRNA exon": "exp232-v4_ncrna_exon-step-4999",
            "TSS / 5′ UTR": "exp232-v4_tss_region_and_utr5-step-4999",
            "Enhancer": "exp232-v4_ccre_non_promoter-step-4999",
            "Background": "exp232-v4_bg-step-4999",
        },
    ),
    Experiment(
        key="annotation_first",
        title="Earlier #517 annotation-first\nphyloP, center-1 projection",
        models={
            "CDS": "exp517-cds-step-4999",
            "3′ UTR": "exp517-utr3-step-4999",
            "ncRNA exon": "exp517-ncrna-step-4999",
            "TSS / 5′ UTR": "exp517-tss_region-step-4999",
            "Enhancer": "exp517-enhancer-step-4999",
        },
    ),
    Experiment(
        key="gpn_uniform",
        title="#517 uniform grid\nGPN-Star-P, center-1 projection",
        models={
            "CDS": "exp517-gpn-uniform-cds-step-4999",
            "3′ UTR": "exp517-gpn-uniform-utr3-step-4999",
            "ncRNA exon": "exp517-gpn-uniform-ncrna-exon-step-4999",
            "TSS / 5′ UTR": "exp517-gpn-uniform-tss-utr5-step-4999",
            "Enhancer": "exp517-gpn-uniform-enhancer-arm-a-step-4999",
            "Background": "exp517-gpn-uniform-background-step-4999",
        },
    ),
    Experiment(
        key="phylop_uniform",
        title="#517 uniform grid\nphyloP, center-1 projection",
        models={
            "CDS": "exp517-phylop-uniform-cds-step-4999",
            "3′ UTR": "exp517-phylop-uniform-utr3-step-4999",
            "ncRNA exon": "exp517-phylop-uniform-ncrna-exon-step-4999",
            "TSS / 5′ UTR": "exp517-phylop-uniform-tss-utr5-step-4999",
            "Enhancer": "exp517-phylop-uniform-enhancer-arm-a-step-4999",
            "Background": "exp517-phylop-uniform-background-step-4999",
        },
    ),
]


def load_metrics() -> pd.DataFrame:
    records: list[dict[str, str | float | int]] = []
    for experiment in EXPERIMENTS:
        for arm, model in experiment.models.items():
            path = f"{S3_PREFIX}/{model}/mendelian_traits.parquet"
            frame = pd.read_parquet(path)
            frame = frame[frame["score_type"].eq(SCORE_TYPE)].set_index("subset")
            for subset, subset_slug in SUBSETS:
                row = frame.loc[subset_slug]
                records.append(
                    {
                        "experiment": experiment.key,
                        "experiment_title": experiment.title.replace("\n", " / "),
                        "model": model,
                        "arm": arm,
                        "subset": subset,
                        "value": float(row["value"]),
                        "se": float(row["se"]),
                        "n_groups": int(row["n_groups"]),
                        "n_rows": int(row["n_rows"]),
                    }
                )
    result = pd.DataFrame.from_records(records)
    expected_rows = sum(len(experiment.models) for experiment in EXPERIMENTS) * len(
        SUBSETS
    )
    assert len(result) == expected_rows
    assert result[["value", "se"]].notna().all().all()
    assert result["value"].between(0, 1).all()
    assert (result["se"] >= 0).all()
    return result


def summarize_diagonals(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, str | float | int | bool]] = []
    summaries: list[dict[str, str | float | int]] = []
    for experiment in EXPERIMENTS:
        frame = metrics[metrics["experiment"].eq(experiment.key)]
        for subset, _ in SUBSETS:
            subset_frame = frame[frame["subset"].eq(subset)].copy()
            home_arm = HOME_ARM[subset]
            home = subset_frame[subset_frame["arm"].eq(home_arm)].iloc[0]
            nonhome = subset_frame[~subset_frame["arm"].eq(home_arm)]
            winner = subset_frame.loc[subset_frame["value"].idxmax()]
            rows.append(
                {
                    "experiment": experiment.key,
                    "subset": subset,
                    "home_arm": home_arm,
                    "home_auprc": float(home["value"]),
                    "home_se": float(home["se"]),
                    "best_nonhome_arm": str(
                        nonhome.loc[nonhome["value"].idxmax(), "arm"]
                    ),
                    "best_nonhome_auprc": float(nonhome["value"].max()),
                    "home_margin": float(home["value"] - nonhome["value"].max()),
                    "row_winner": str(winner["arm"]),
                    "home_wins": bool(winner["arm"] == home_arm),
                }
            )
        experiment_rows = [row for row in rows if row["experiment"] == experiment.key]
        margins = [float(row["home_margin"]) for row in experiment_rows]
        home_values = [float(row["home_auprc"]) for row in experiment_rows]
        background_wins = sum(
            str(row["row_winner"]) == "Background" for row in experiment_rows
        )
        summaries.append(
            {
                "experiment": experiment.key,
                "diagonal_wins": sum(bool(row["home_wins"]) for row in experiment_rows),
                "total_home_rows": len(experiment_rows),
                "mean_home_auprc": float(np.mean(home_values)),
                "mean_home_margin": float(np.mean(margins)),
                "minimum_home_margin": float(np.min(margins)),
                "background_wins": background_wins,
            }
        )
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(summaries)


def plot_diagonals(metrics: pd.DataFrame, output_path: Path) -> None:
    arm_order = [arm for arm, _ in ARMS]
    subset_order = [subset for subset, _ in SUBSETS]
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams["svg.fonttype"] = "none"
    cmap = mpl.colormaps["Reds"].copy()
    cmap.set_bad("#eeeeee")

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 16.0))
    colorbar_ax = fig.add_axes((0.925, 0.25, 0.016, 0.50))
    for index, (ax, experiment) in enumerate(zip(axes.flat, EXPERIMENTS)):
        frame = metrics[metrics["experiment"].eq(experiment.key)]
        values = frame.pivot(index="subset", columns="arm", values="value").reindex(
            index=subset_order, columns=arm_order
        )
        errors = frame.pivot(index="subset", columns="arm", values="se").reindex(
            index=subset_order, columns=arm_order
        )
        sns.heatmap(
            values,
            ax=ax,
            cmap=cmap,
            vmin=0.10,
            vmax=0.56,
            square=True,
            linewidths=0.7,
            linecolor="white",
            cbar=index == len(EXPERIMENTS) - 1,
            cbar_ax=colorbar_ax if index == len(EXPERIMENTS) - 1 else None,
            cbar_kws={"label": "AUPRC"},
        )
        for row_index, subset in enumerate(subset_order):
            row_values = values.loc[subset]
            available = row_values.dropna()
            winner_arm = str(available.idxmax())
            for column_index, arm in enumerate(arm_order):
                value = values.loc[subset, arm]
                error = errors.loc[subset, arm]
                if pd.isna(value):
                    label = "—"
                    color = "#666666"
                else:
                    label = f"{value:.3f}\n±{error:.3f}"
                    color = "white" if value >= 0.34 else "#202020"
                ax.text(
                    column_index + 0.5,
                    row_index + 0.5,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7.0,
                )
                if HOME_ARM[subset] == arm:
                    ax.add_patch(
                        Rectangle(
                            (column_index, row_index),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            linewidth=2.0,
                        )
                    )
            winner_index = arm_order.index(winner_arm)
            ax.scatter(
                winner_index + 0.80,
                row_index + 0.20,
                marker="*",
                s=80,
                color="#7CFC00",
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
        ax.set_title(experiment.title, pad=10)
        ax.set_xlabel("Training arm")
        ax.set_ylabel("Mendelian variant subset")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    legend_handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2.0),
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
    fig.legend(
        legend_handles,
        ["Matched training scope", "Row maximum"],
        title="Markers",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Historical Mendelian specialist diagonals", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.09, hspace=0.28)
    fig.savefig(output_path, bbox_inches="tight", metadata={"Date": None})
    fig.savefig(output_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    home_rows, summaries = summarize_diagonals(metrics)
    metrics.to_csv(OUTPUT_DIR / "issue517_historical_diagonals_data.csv", index=False)
    home_rows.to_csv(
        OUTPUT_DIR / "issue517_historical_diagonals_home_rows.csv", index=False
    )
    summaries.to_csv(
        OUTPUT_DIR / "issue517_historical_diagonals_summary.csv", index=False
    )
    plot_diagonals(
        metrics, OUTPUT_DIR / "issue517_historical_diagonals_comparison.svg"
    )


if __name__ == "__main__":
    main()
