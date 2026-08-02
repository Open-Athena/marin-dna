"""Plot the audited Stage-9 repeat-aware Mendelian label summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

STRATA = ("all", "focal_repeat", "near_repeat", "repeat_free_window")
STRATUM_LABELS = {
    "all": "All",
    "focal_repeat": "Focal repeat",
    "near_repeat": "Repeat elsewhere",
    "repeat_free_window": "Repeat-free",
}
COLORS = {
    "all": "#4c78a8",
    "focal_repeat": "#e45756",
    "near_repeat": "#f2cf5b",
    "repeat_free_window": "#54a24b",
}
ORIENTATIONS = ("forward", "reverse_complement")
ORIENTATION_LABELS = {"forward": "FWD", "reverse_complement": "RC"}
LINESTYLES = {"forward": "-", "reverse_complement": "--"}
MARKERS = {"forward": "o", "reverse_complement": "s"}


def load_plot_tables(root: Path) -> dict[str, pl.DataFrame]:
    tables = {
        "counts": pl.read_parquet(root / "stratum_target_counts.parquet").filter(
            pl.col("target") == "overall"
        ),
        "summary": pl.read_parquet(root / "target_summary.parquet").filter(
            pl.col("target") == "overall"
        ),
        "retention": pl.read_parquet(root / "repeat_free_retention.parquet").filter(
            pl.col("target") == "overall"
        ),
        "feature": pl.read_parquet(root / "feature9086.parquet").filter(
            (pl.col("target") == "overall") & (pl.col("response") == "abs_delta")
        ),
    }
    assert tables["counts"].height == len(STRATA)
    assert tables["summary"].height == 3 * 2 * 2 * len(STRATA)
    assert tables["retention"].height == 3 * 2 * 2
    assert tables["feature"].height == len(STRATA) * 2
    return tables


def plot(root: Path, output_dir: Path) -> tuple[Path, Path]:
    tables = load_plot_tables(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    counts = tables["counts"]
    prevalence = [
        float(counts.filter(pl.col("stratum") == stratum)["prevalence"].item())
        for stratum in STRATA
    ]
    axes[0, 0].bar(
        range(len(STRATA)),
        np.asarray(prevalence) * 100,
        color=[COLORS[stratum] for stratum in STRATA],
    )
    axes[0, 0].axhline(10, color="black", linewidth=1, linestyle=":")
    axes[0, 0].set_xticks(
        range(len(STRATA)), [STRATUM_LABELS[stratum] for stratum in STRATA], rotation=18
    )
    axes[0, 0].set_ylabel("Pathogenic label prevalence (%)")
    axes[0, 0].set_title("A. Repeat context is label-imbalanced")
    for index, value in enumerate(prevalence):
        axes[0, 0].text(index, value * 100 + 0.35, f"{value:.1%}", ha="center")

    summary = tables["summary"].filter(pl.col("response") == "abs_delta")
    for stratum in STRATA:
        for orientation in ORIENTATIONS:
            current = summary.filter(
                (pl.col("repeat_stratum") == stratum)
                & (pl.col("orientation") == orientation)
            ).sort("block")
            axes[0, 1].plot(
                current["block"],
                100
                * current["discoveries"].to_numpy()
                / current["eligible_features"].to_numpy(),
                color=COLORS[stratum],
                linestyle=LINESTYLES[orientation],
                marker=MARKERS[orientation],
                label=f"{STRATUM_LABELS[stratum]} {ORIENTATION_LABELS[orientation]}",
            )
    axes[0, 1].set_xticks([1, 10, 19])
    axes[0, 1].set_xlabel("Model block")
    axes[0, 1].set_ylabel("Eligible |Δ| features discovered (%)")
    axes[0, 1].set_title("B. Block 19 dominates with or without repeats")
    axes[0, 1].legend(fontsize=7, ncol=2, frameon=False)

    retention = tables["retention"]
    response_colors = {"abs_delta": "#4c78a8", "delta": "#b279a2"}
    response_labels = {"abs_delta": "|Δ|", "delta": "signed Δ"}
    for response in ("abs_delta", "delta"):
        for orientation in ORIENTATIONS:
            current = retention.filter(
                (pl.col("response") == response)
                & (pl.col("orientation") == orientation)
            ).sort("block")
            axes[1, 0].plot(
                current["block"],
                100 * current["retention_fraction"].to_numpy(),
                color=response_colors[response],
                linestyle=LINESTYLES[orientation],
                marker=MARKERS[orientation],
                label=f"{response_labels[response]} {ORIENTATION_LABELS[orientation]}",
            )
    axes[1, 0].set_xticks([1, 10, 19])
    axes[1, 0].set_ylim(-3, 103)
    axes[1, 0].set_xlabel("Model block")
    axes[1, 0].set_ylabel("Global discoveries retained repeat-free (%)")
    axes[1, 0].set_title("C. Most block-19 discoveries persist repeat-free")
    axes[1, 0].legend(fontsize=8, frameon=False)

    feature = tables["feature"]
    for orientation in ORIENTATIONS:
        current = feature.filter(pl.col("orientation") == orientation)
        lifts = []
        for stratum in STRATA:
            row = current.filter(pl.col("repeat_stratum") == stratum)
            lifts.append(float(row["best_auprc"].item() / row["prevalence"].item()))
        axes[1, 1].plot(
            range(len(STRATA)),
            lifts,
            color="#7a5195",
            linestyle=LINESTYLES[orientation],
            marker=MARKERS[orientation],
            label=ORIENTATION_LABELS[orientation],
        )
    axes[1, 1].axhline(1, color="black", linewidth=1, linestyle=":")
    axes[1, 1].set_xticks(
        range(len(STRATA)), [STRATUM_LABELS[stratum] for stratum in STRATA], rotation=18
    )
    axes[1, 1].set_ylabel("Feature 9086 AUPRC / prevalence")
    axes[1, 1].set_title("D. Leading feature 9086 is not repeat-confounded")
    axes[1, 1].legend(frameon=False)

    figure.suptitle(
        "Repeat context does not explain the dominant block-19 Mendelian label signal",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "Discoveries require Welch and Mann–Whitney BH q<0.05; FWD and RC are separate.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    png = output_dir / "repeat_label_sensitivity.png"
    svg = output_dir / "repeat_label_sensitivity.svg"
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.analysis_root, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
