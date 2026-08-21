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


def save_evidence(fig: plt.Figure, stem: str) -> None:
    """Save an evidence figure for documentation and local review."""

    fig.savefig(FIGURES / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight")
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
    axis.bar(
        x - width / 2, frame["left_l1"], width, label="Left flank", color="#0072B2"
    )
    axis.bar(
        x + width / 2, frame["right_l1"], width, label="Right flank", color="#E69F00"
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean L1 change in target distribution")
    axis.set_title(
        "MNTP acquires bilateral context use; causal controls remain right-blind"
    )
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
            stability[stability["dataset"] == dataset]
            .set_index("condition")
            .loc[condition_order]
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


def plot_validation_trajectories() -> None:
    """Plot archived validation histories with independent checkpoint recomputation."""

    logged = pd.read_csv(ROOT / "training-validation.csv")
    audit = pd.read_csv(ROOT / "audit" / "checkpoint-loss-audit.csv")
    colors = {
        "transferred_mntp": "#E45756",
        "scratch_mntp": "#54A24B",
        "clm_continuation": "#4C78A8",
    }
    labels = {
        "transferred_mntp": "Transferred MNTP",
        "scratch_mntp": "Scratch MNTP",
        "clm_continuation": "Continued CLM",
    }
    panels = (
        (
            "Diffusion-mask MNTP",
            "diffusion",
            (
                (
                    "transferred_mntp",
                    "full_attention_no_adaptation",
                    "transferred_diffusion_loss",
                ),
                ("scratch_mntp", "scratch_mntp", "scratch_diffusion_loss"),
            ),
        ),
        (
            "Single-mask MNTP",
            "single",
            (
                (
                    "transferred_mntp",
                    "full_attention_no_adaptation",
                    "transferred_single_mask_loss",
                ),
                ("scratch_mntp", "scratch_mntp", "scratch_single_mask_loss"),
            ),
        ),
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for axis, (title, mode, series) in zip(axes[:2], panels, strict=True):
        for arm, initial_arm, column in series:
            initial = audit[
                (audit["arm"] == initial_arm)
                & (audit["validation_mode"] == mode)
                & (audit["step"] == 0)
            ]
            if len(initial) != 1:
                raise ValueError(
                    f"expected one {initial_arm} {mode} step-zero row, found {len(initial)}"
                )
            x = np.concatenate(([0], logged["step"].to_numpy()))
            y = np.concatenate(([initial.iloc[0]["loss"]], logged[column].to_numpy()))
            axis.plot(
                x,
                y,
                color=colors[arm],
                linewidth=1.8,
                label=labels[arm],
            )
            recomputed = audit[
                (audit["arm"].isin((initial_arm, arm)))
                & (audit["validation_mode"] == mode)
            ].sort_values("step")
            axis.scatter(
                recomputed["step"],
                recomputed["loss"],
                facecolors="white",
                edgecolors=colors[arm],
                linewidths=1.2,
                s=32,
                zorder=3,
            )
        axis.set_title(title)
        axis.legend(title="Arm", frameon=False)

    causal = axes[2]
    causal_color = colors["clm_continuation"]
    clm_initial = audit[
        (audit["arm"] == "clm_continuation")
        & (audit["validation_mode"] == "causal")
        & (audit["step"] == 0)
        & (audit["kind"] == "replay")
    ]
    if len(clm_initial) != 1:
        raise ValueError(f"expected one CLM save/reload row, found {len(clm_initial)}")
    causal.plot(
        np.concatenate(([0], logged["step"].to_numpy())),
        np.concatenate(
            ([clm_initial.iloc[0]["loss"]], logged["clm_diffusion_loss"].to_numpy())
        ),
        color=causal_color,
        linewidth=1.8,
        label=labels["clm_continuation"],
    )
    recomputed_clm = audit[
        (audit["arm"] == "clm_continuation")
        & (audit["validation_mode"] == "causal")
        & ~((audit["step"] == 400) & (audit["kind"] == "lightning"))
    ].sort_values("step")
    causal.scatter(
        recomputed_clm["step"],
        recomputed_clm["loss"],
        facecolors="white",
        edgecolors=causal_color,
        linewidths=1.2,
        s=32,
        zorder=3,
    )
    source_direct = audit[
        (audit["arm"] == "source_clm")
        & (audit["validation_mode"] == "causal")
        & (audit["step"] == 0)
    ]
    if len(source_direct) != 1:
        raise ValueError(f"expected one direct source row, found {len(source_direct)}")
    causal.scatter(
        [0],
        source_direct["loss"],
        marker="o",
        facecolors="none",
        edgecolors="#888888",
        linewidths=1.8,
        s=90,
        zorder=4,
        label="Source save/reload",
    )
    causal.scatter(
        [0],
        source_direct["loss"],
        marker="D",
        color="#222222",
        s=25,
        zorder=5,
        label="Source direct",
    )
    causal.set_title("Causal CLM")
    causal.legend(title="Arm or control", frameon=False)

    for axis in axes:
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Weighted cross-entropy")
        axis.grid(alpha=0.25)
        axis.set_box_aspect(1)
    figure.suptitle("Validation trajectories reproduce at independent checkpoints")
    figure.subplots_adjust(top=0.82, bottom=0.15, wspace=0.3)
    save_evidence(figure, "validation-trajectories")


def plot_auprc_evidence() -> None:
    """Plot primary odd-autosome/X AUPRC trajectories with uncertainty."""

    metrics = pd.read_csv(ROOT / "audit" / "checkpoint-auprc.csv")
    selected = metrics[metrics["orientation"] == "protocol_fwd_rc"]
    datasets = ("mendelian_traits", "complex_traits", "sge")
    titles = ("Mendelian traits", "Complex traits", "SGE")
    styles = {
        "Continued CLM replay": ("#4C78A8", "o", "-"),
        "Continued CLM original": ("#4C78A8", "s", "--"),
        "Transferred MNTP": ("#E45756", "o", "-"),
        "Scratch MNTP": ("#54A24B", "^", "-"),
        "Source CLM direct": ("#222222", "D", "None"),
        "Source CLM save/reload": ("#888888", "x", "None"),
    }
    display_labels = {
        "Continued CLM replay": "Continued CLM replay",
        "Continued CLM original": "Continued CLM archived",
        "Transferred MNTP": "Transferred MNTP",
        "Scratch MNTP": "Scratch MNTP",
        "Source CLM direct": "Source CLM direct",
        "Source CLM save/reload": "Source CLM save/reload",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    for axis, dataset, title in zip(axes, datasets, titles, strict=True):
        cell = selected[selected["dataset"] == dataset]
        for series, group in cell.groupby("plot_series", sort=False):
            color, marker, linestyle = styles[series]
            group = group.sort_values("step")
            axis.errorbar(
                group["step"],
                group["auprc"],
                yerr=group["se"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                capsize=2,
                label=display_labels[series],
            )
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("AUPRC")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_box_aspect(1)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        title="Checkpoint series",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=3,
    )
    figure.suptitle("Primary odd-autosome/X VEP trajectories")
    figure.subplots_adjust(top=0.7, bottom=0.13, wspace=0.3)
    save_evidence(figure, "auprc-trajectories-audited")


def plot_knowledge_summary() -> None:
    """Render the accepted issue-479 interpretation as a compact lead figure."""

    palette = plt.get_cmap("tab10").colors

    validation_rows = pd.read_csv(ROOT / "training-validation.csv").query(
        "step == 1000"
    )
    if len(validation_rows) != 1:
        raise ValueError(
            f"expected one step-1,000 validation row, found {len(validation_rows)}"
        )
    validation = validation_rows.iloc[0]
    loss_modes = ("Diffusion mask", "Single mask")
    loss_advantage = np.array(
        [
            validation["scratch_diffusion_loss"]
            - validation["transferred_diffusion_loss"],
            validation["scratch_single_mask_loss"]
            - validation["transferred_single_mask_loss"],
        ]
    )

    dependency = pd.read_csv(ROOT / "audit" / "final-checkpoint-dependency-summary.csv")
    dependency_labels = {
        "transferred_mntp": "Transferred MNTP",
        "scratch_mntp": "Scratch MNTP",
        "clm_continuation": "Continued CLM",
    }
    dependency_colors = {
        "transferred_mntp": palette[0],
        "scratch_mntp": palette[1],
        "clm_continuation": palette[2],
    }

    endpoints = pd.read_csv(ROOT / "primary-endpoints.csv")
    endpoint_order = ("mendelian_traits", "complex_traits", "sge")
    endpoint_labels = ("Mendelian", "Complex traits", "SGE")
    score_labels = {
        "source_clm_avg": "Source CLM",
        "transferred_mntp_fwd": "Transferred MNTP FWD",
    }

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

    axes[0].bar(loss_modes, loss_advantage, color=palette[0])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Scratch − transferred cross-entropy")
    axes[0].set_title("Transferred validation advantage")
    axes[0].grid(axis="y", alpha=0.25)

    context_order = ("past_context", "future_context")
    context_labels = ("Past context", "Future context")
    x = np.arange(len(context_order))
    width = 0.24
    for index, arm in enumerate(dependency_labels):
        cell = dependency[dependency["arm"] == arm].set_index("region")
        axes[1].bar(
            x + (index - 1) * width,
            cell.loc[list(context_order), "mean_dependency"],
            width,
            color=dependency_colors[arm],
            label=dependency_labels[arm],
        )
    axes[1].set_xticks(x, context_labels)
    axes[1].set_ylabel("Mean L∞ log-probability change")
    axes[1].set_title("Final dependency by direction")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(title="Checkpoint", frameon=False)

    x = np.arange(len(endpoint_order))
    offsets = (-0.08, 0.08)
    for offset, (score_type, label), color in zip(
        offsets,
        score_labels.items(),
        (palette[2], palette[0]),
        strict=True,
    ):
        cell = endpoints[endpoints["score_type"] == score_type].set_index("dataset")
        cell = cell.loc[list(endpoint_order)]
        axes[2].errorbar(
            x + offset,
            cell["auprc"],
            yerr=cell["se"],
            color=color,
            fmt="o",
            capsize=3,
            label=label,
        )
    axes[2].set_xticks(x, endpoint_labels)
    axes[2].set_ylabel("AUPRC (±1 SE)")
    axes[2].set_title("Source-relative VEP")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(title="Checkpoint", frameon=False)

    for axis in axes:
        axis.set_box_aspect(1)
    figure.suptitle(
        "1,000-step MNTP conversion: transfer signal, bilateral context, lower VEP"
    )
    save(figure, "issue-479-lead.svg")


def main() -> None:
    plot_validation()
    plot_primary_vep()
    plot_context()
    plot_context_window()
    plot_nucleotide_dependency()
    plot_validation_trajectories()
    plot_auprc_evidence()
    plot_knowledge_summary()


if __name__ == "__main__":
    main()
