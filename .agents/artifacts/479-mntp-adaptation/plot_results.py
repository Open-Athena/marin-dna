"""Render the compact issue #479 result figures from committed tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "source_clm_avg": "#0072B2",
    "transferred_mntp_fwd": "#D55E00",
    "scratch_mntp_fwd": "#009E73",
    "clm_continuation_avg": "#56B4E9",
    "transferred_mntp_avg": "#CC79A7",
    "transferred": "#D55E00",
    "scratch": "#009E73",
}
LABELS = {
    "source_clm_avg": "Source CLM\nFWD+RC",
    "transferred_mntp_fwd": "Transferred\nMNTP FWD",
    "scratch_mntp_fwd": "Scratch\nMNTP FWD",
    "clm_continuation_avg": "Continued CLM\nFWD+RC",
    "transferred_mntp_avg": "Transferred\nFWD+RC",
}
CONDITIONS = {
    "left_context_ablated": "Left flank\nablated",
    "right_context_ablated": "Right flank\nablated",
    "window_shift_upstream_64": "Window 64 bp\nupstream",
    "window_shift_downstream_64": "Window 64 bp\ndownstream",
}
DATASETS = {
    "mendelian_traits": "Mendelian",
    "complex_traits": "Complex traits",
    "sge": "SGE",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def plot_validation() -> None:
    frame = pd.read_csv(ROOT / "training-validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for arm in ("transferred", "scratch"):
        axes[0].plot(
            frame["step"],
            frame[f"{arm}_diffusion_loss"],
            marker="o",
            color=COLORS[arm],
            label=arm.capitalize(),
        )
        axes[1].plot(
            frame["step"],
            frame[f"{arm}_single_mask_loss"],
            marker="o",
            color=COLORS[arm],
            label=arm.capitalize(),
        )
    axes[0].set_title("Diffusion-mask validation")
    axes[1].set_title("Single-mask validation")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Weighted cross-entropy")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Transferred MNTP narrowly beats scratch at step 1,000 (one seed)")
    fig.text(
        0.5,
        -0.02,
        "Deterministic fixed validation; lower is better. Lines show one run per arm, not uncertainty.",
        ha="center",
        fontsize=9,
    )
    save(fig, "validation-loss.svg")


def plot_primary_vep() -> None:
    frame = pd.read_csv(ROOT / "primary-endpoints.csv")
    order = list(LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        cell = frame[frame["dataset"] == dataset].set_index("score_type").loc[order]
        x = np.arange(len(order))
        axis.bar(
            x,
            cell["auprc"],
            yerr=cell["se"],
            capsize=3,
            color=[COLORS[name] for name in order],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_title(DATASETS[dataset])
        axis.set_xticks(x, [LABELS[name] for name in order], rotation=28, ha="right")
        axis.set_ylim(0, 0.45)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUPRC (bar ± 1 SE)")
    fig.suptitle("Transferred MNTP does not improve a primary VEP endpoint")
    save(fig, "primary-vep.svg")


def plot_context() -> None:
    frame = pd.read_parquet(ROOT / "vep" / "context-probes.parquet")
    order = [
        "source_clm",
        "full_attention_no_adaptation",
        "transferred_mntp",
        "scratch_mntp",
        "clm_continuation",
    ]
    labels = [
        "Source CLM",
        "Full attention\nno adaptation",
        "Transferred\nMNTP",
        "Scratch\nMNTP",
        "Continued\nCLM",
    ]
    frame = frame.set_index("arm").loc[order]
    x = np.arange(len(order))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(x - width / 2, frame["left_l1"], width, label="Left flank", color="#0072B2")
    axis.bar(x + width / 2, frame["right_l1"], width, label="Right flank", color="#E69F00")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean L1 change in target distribution")
    axis.set_title("MNTP acquires bilateral context use; causal controls remain right-blind")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(fig, "context-probes.svg")


def plot_context_window() -> None:
    stability = pd.read_parquet(ROOT / "context-window" / "stability.parquet")
    stability = stability[stability["condition"] != "centered_full"].copy()
    condition_order = list(CONDITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(condition_order))
    offsets = np.linspace(-0.25, 0.25, len(DATASETS))
    for offset, (dataset, label) in zip(offsets, DATASETS.items(), strict=True):
        cell = (
            stability[stability["dataset"] == dataset].set_index("condition").loc[condition_order]
        )
        axes[0].plot(
            x + offset,
            cell["spearman_vs_centered"],
            marker="o",
            linestyle="none",
            label=label,
        )
        axes[1].plot(
            x + offset,
            cell["mean_absolute_llr_change"],
            marker="o",
            linestyle="none",
            label=label,
        )
    for axis in axes:
        axis.set_xticks(x, [CONDITIONS[name] for name in condition_order])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Spearman ρ vs centered score")
    axes[0].set_ylim(0.65, 1.01)
    axes[0].set_title("Ranking stability")
    axes[1].set_ylabel("Mean absolute raw-LLR change")
    axes[1].set_title("Score sensitivity")
    axes[1].legend(frameon=False, loc="center right")
    fig.suptitle("Both flanks matter; ±64-bp window placement is stable")
    save(fig, "context-window-stability.svg")


def plot_nucleotide_dependency() -> None:
    summary = pd.DataFrame(
        json.loads((ROOT / "nucleotide-dependency" / "summary.json").read_text())
    )
    fig, axis = plt.subplots(figsize=(8, 3.8))
    axis.bar(
        summary["locus"],
        summary["off_diagonal_spearman"],
        color="#009E73",
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_ylim(0.90, 1.00)
    axis.set_ylabel("Off-diagonal Spearman ρ")
    axis.set_title("Single-orientation and FWD+RC dependency maps are similar")
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(summary["off_diagonal_spearman"]):
        axis.text(index, value + 0.0006, f"{value:.3f}", ha="center", fontsize=9)
    save(fig, "nucleotide-dependency-correlation.svg")


def main() -> None:
    plot_validation()
    plot_primary_vep()
    plot_context()
    plot_context_window()
    plot_nucleotide_dependency()


if __name__ == "__main__":
    main()
