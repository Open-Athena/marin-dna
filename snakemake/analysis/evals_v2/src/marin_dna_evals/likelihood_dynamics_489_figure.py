"""Research figures for likelihood dynamics through m1.3 (issue #489)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCOPE_ORDER = ("global", "cds", "upstream", "downstream", "ncrna", "enhancer")
REGION_ORDER = ("cds", "upstream", "downstream", "ncrna", "enhancer")
SCOPE_LABELS = {
    "global": "Global",
    "cds": "CDS",
    "upstream": "Upstream",
    "downstream": "Downstream",
    "ncrna": "ncRNA",
    "enhancer": "Enhancer",
}
STATISTIC_LABELS = {"loss": "Loss", "entropy": "Entropy"}
GROUP_LABELS = {
    "low_to_low": "Low to low",
    "low_to_high": "Low to high",
    "high_to_low": "High to low",
    "high_to_high": "High to high",
}


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _finish(
    figure: plt.Figure,
    output_path: str | Path,
    *,
    legend_handles: list[object],
    legend_labels: list[str],
    legend_title: str,
) -> None:
    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=max(1, len(legend_labels)),
            frameon=False,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(_prepare_output(output_path), bbox_inches="tight")
    plt.close(figure)


def _six_panel() -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    for axis in axes.flat:
        axis.set_box_aspect(1)
    return figure, axes


def plot_conservation_auprc_489(
    metrics_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot exact conservation AUPRC against cumulative training tokens."""
    frame = pd.read_parquet(metrics_path)
    palette = dict(
        zip(
            ("loss", "entropy"),
            sns.color_palette(n_colors=2),
            strict=True,
        )
    )
    sns.set_theme(style="whitegrid")
    figure, axes = _six_panel()
    handles: list[object] = []
    labels: list[str] = []
    for axis, scope in zip(axes.flat, SCOPE_ORDER, strict=True):
        subset = frame[frame["scope"] == scope].sort_values("checkpoint_order")
        assert len(subset) > 0
        for statistic in ("loss", "entropy"):
            values = subset[subset["statistic"] == statistic]
            (line,) = axis.plot(
                values["cumulative_tokens"] / 1e9,
                values["auprc"],
                marker="o",
                color=palette[statistic],
                label=STATISTIC_LABELS[statistic],
            )
            if scope == "global":
                handles.append(line)
                labels.append(STATISTIC_LABELS[statistic])
        prevalence = float(subset["prevalence"].iloc[0])
        axis.axhline(
            prevalence,
            color="0.45",
            linestyle="--",
            linewidth=1,
        )
        axis.set_title(SCOPE_LABELS[scope])
        axis.set_xlabel("Cumulative training tokens (billions)")
        axis.set_ylabel("Conservation AUPRC")
        axis.set_ylim(0, 1)
    figure.suptitle("Conservation ranking across the m1-to-m1.3 lineage")
    _finish(
        figure,
        output_path,
        legend_handles=handles,
        legend_labels=labels,
        legend_title="Score",
    )


def plot_trajectory_groups_489(
    trajectory_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot mean loss for the four earliest/terminal threshold groups."""
    frame = pd.read_parquet(trajectory_path)
    group_order = tuple(GROUP_LABELS)
    palette = dict(zip(group_order, sns.color_palette(n_colors=4), strict=True))
    sns.set_theme(style="whitegrid")
    figure, axes = _six_panel()
    handles: list[object] = []
    labels: list[str] = []
    for axis, scope in zip(axes.flat, SCOPE_ORDER, strict=True):
        subset = frame[frame["scope"] == scope]
        for group in group_order:
            values = subset[subset["group"] == group].sort_values("checkpoint_order")
            if values.empty:
                continue
            x = values["cumulative_tokens"].to_numpy(dtype=float) / 1e9
            (line,) = axis.plot(
                x,
                values["mean"],
                marker="o",
                color=palette[group],
                label=GROUP_LABELS[group],
            )
            axis.fill_between(
                x,
                values["ci_low"],
                values["ci_high"],
                color=palette[group],
                alpha=0.15,
                linewidth=0,
            )
            if scope == "global":
                handles.append(line)
                labels.append(GROUP_LABELS[group])
        axis.set_title(SCOPE_LABELS[scope])
        axis.set_xlabel("Cumulative training tokens (billions)")
        axis.set_ylabel("Mean loss (nats/base)")
    figure.suptitle("Loss trajectories by earliest and terminal state")
    _finish(
        figure,
        output_path,
        legend_handles=handles,
        legend_labels=labels,
        legend_title="Trajectory group",
    )


def plot_selection_jaccard_489(
    overlap_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot exact overlap of region-specific lowest-score deciles."""
    frame = pd.read_parquet(overlap_path).copy()
    frame["pair_order"] = np.where(
        frame["comparison"] == "endpoint",
        len(frame["checkpoint_order_from"].unique()) + 1,
        frame["checkpoint_order_from"],
    )
    pair_order = (
        frame[
            [
                "pair_order",
                "checkpoint_order_from",
                "checkpoint_order_to",
            ]
        ]
        .drop_duplicates()
        .sort_values("pair_order")
    )
    pair_labels = [
        f"{int(row.checkpoint_order_from) + 1}-{int(row.checkpoint_order_to) + 1}"
        for row in pair_order.itertuples(index=False)
    ]
    palette = dict(
        zip(SCOPE_ORDER, sns.color_palette(n_colors=len(SCOPE_ORDER)), strict=True)
    )
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), squeeze=False)
    handles: list[object] = []
    labels: list[str] = []
    for axis, statistic in zip(axes.flat, ("loss", "entropy"), strict=True):
        for scope in SCOPE_ORDER:
            values = frame[
                (frame["statistic"] == statistic) & (frame["scope"] == scope)
            ].sort_values("pair_order")
            (line,) = axis.plot(
                np.arange(len(values)),
                values["jaccard"],
                marker="o",
                color=palette[scope],
                label=SCOPE_LABELS[scope],
            )
            if statistic == "loss":
                handles.append(line)
                labels.append(SCOPE_LABELS[scope])
        axis.set_xticks(np.arange(len(pair_labels)), pair_labels)
        axis.set_title(STATISTIC_LABELS[statistic])
        axis.set_xlabel("Checkpoint pair")
        axis.set_ylabel("Lowest-decile Jaccard")
        axis.set_ylim(0, 1)
        axis.set_box_aspect(1)
    figure.suptitle("Stability of the lowest-score 10% within each region")
    _finish(
        figure,
        output_path,
        legend_handles=handles,
        legend_labels=labels,
        legend_title="Scope",
    )


def plot_future_loss_deciles_489(
    decile_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot terminal loss reduction by current region-specific loss decile."""
    frame = pd.read_parquet(decile_path)
    frame = frame[frame["horizon"] == "terminal"]
    orders = sorted(frame["current_checkpoint_order"].unique())
    palette = dict(zip(orders, sns.color_palette(n_colors=len(orders)), strict=True))
    sns.set_theme(style="whitegrid")
    figure, axes = _six_panel()
    handles: list[object] = []
    labels: list[str] = []
    for axis, scope in zip(axes.flat, SCOPE_ORDER, strict=True):
        subset = frame[frame["scope"] == scope]
        for order in orders:
            values = subset[subset["current_checkpoint_order"] == order].sort_values(
                "current_loss_bin"
            )
            x = values["current_loss_bin"].to_numpy(dtype=float)
            (line,) = axis.plot(
                x,
                values["mean"],
                marker="o",
                color=palette[order],
                label=f"Checkpoint {order + 1}",
            )
            axis.fill_between(
                x,
                values["ci_low"],
                values["ci_high"],
                color=palette[order],
                alpha=0.12,
                linewidth=0,
            )
            if scope == "global":
                handles.append(line)
                labels.append(f"Checkpoint {order + 1}")
        axis.axhline(0, color="0.45", linewidth=1)
        axis.set_title(SCOPE_LABELS[scope])
        axis.set_xlabel("Current-loss decile (low to high)")
        axis.set_ylabel("Loss reduction to terminal (nats/base)")
        axis.set_xticks(range(1, 11))
    figure.suptitle("Future loss reduction by current loss")
    _finish(
        figure,
        output_path,
        legend_handles=handles,
        legend_labels=labels,
        legend_title="Current model",
    )


def plot_score_distributions_489(
    distributions_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot block-bootstrap mean score trajectories by conservation label."""
    frame = pd.read_parquet(distributions_path)
    frame = frame[frame["scope"].isin(REGION_ORDER)]
    palette = dict(zip((False, True), sns.color_palette(n_colors=2), strict=True))
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 5, figsize=(15, 6), squeeze=False)
    handles: list[object] = []
    labels: list[str] = []
    for row, statistic in enumerate(("loss", "entropy")):
        for column, scope in enumerate(REGION_ORDER):
            axis = axes[row, column]
            subset = frame[
                (frame["scope"] == scope) & (frame["statistic"] == statistic)
            ]
            for conserved in (False, True):
                values = subset[subset["conserved"] == conserved].sort_values(
                    "checkpoint_order"
                )
                x = values["cumulative_tokens"].to_numpy(dtype=float) / 1e9
                label = "Conserved" if conserved else "Other"
                (line,) = axis.plot(
                    x,
                    values["mean"],
                    marker="o",
                    color=palette[conserved],
                    label=label,
                )
                axis.fill_between(
                    x,
                    values["ci_low"],
                    values["ci_high"],
                    color=palette[conserved],
                    alpha=0.15,
                    linewidth=0,
                )
                if row == 0 and column == 0:
                    handles.append(line)
                    labels.append(label)
            axis.set_title(
                f"{SCOPE_LABELS[scope]}: {STATISTIC_LABELS[statistic].lower()}"
            )
            axis.set_xlabel("Training tokens (billions)" if row == 1 else "")
            axis.set_ylabel(
                (
                    "Mean loss (nats/base)"
                    if statistic == "loss"
                    else "Mean entropy (nats)"
                )
                if column == 0
                else ""
            )
            axis.set_box_aspect(1)
    figure.suptitle("Score trajectories by conservation label")
    _finish(
        figure,
        output_path,
        legend_handles=handles,
        legend_labels=labels,
        legend_title="Case-derived label",
    )


def plot_controlled_conservation_489(
    controlled_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot adjusted conserved-minus-other contrasts for negative scores."""
    frame = pd.read_parquet(controlled_path)
    frame = frame[frame["term"] == "conserved"]
    palette = dict(zip(("loss", "entropy"), sns.color_palette(n_colors=2), strict=True))
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 5, figsize=(15, 6), squeeze=False)
    handles: list[object] = []
    labels: list[str] = []
    for row, statistic in enumerate(("loss", "entropy")):
        for column, scope in enumerate(REGION_ORDER):
            axis = axes[row, column]
            values = frame[
                (frame["scope"] == scope) & (frame["statistic"] == statistic)
            ].sort_values("checkpoint_order")
            x = values["cumulative_tokens"].to_numpy(dtype=float) / 1e9
            (line,) = axis.plot(
                x,
                values["estimate"],
                marker="o",
                color=palette[statistic],
                label=STATISTIC_LABELS[statistic],
            )
            axis.fill_between(
                x,
                values["ci_low"],
                values["ci_high"],
                color=palette[statistic],
                alpha=0.15,
                linewidth=0,
            )
            axis.axhline(0, color="0.45", linewidth=1)
            axis.set_title(
                f"{SCOPE_LABELS[scope]}: {STATISTIC_LABELS[statistic].lower()}"
            )
            axis.set_xlabel("Training tokens (billions)" if row == 1 else "")
            axis.set_ylabel("Adjusted contrast (nats)" if column == 0 else "")
            axis.set_box_aspect(1)
            if column == 0:
                handles.append(line)
                labels.append(STATISTIC_LABELS[statistic])
    figure.suptitle("Conservation contrast after GC, 7-mer, and position controls")
    _finish(
        figure,
        output_path,
        legend_handles=handles,
        legend_labels=labels,
        legend_title="Negative score",
    )
