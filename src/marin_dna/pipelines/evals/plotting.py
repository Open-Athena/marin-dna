"""Plotting utilities for visualizing evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics_vs_step(
    metrics_df: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """Plot metric values vs training step for a specific model.

    Creates a grid of subplots where each panel shows a (dataset, subset) combination.
    Each panel has separate lines for each scoring method (minus_llr, abs_llr).

    Args:
        metrics_df: DataFrame with columns [step, dataset, metric, score_type, subset, value].
        model_name: Name of the model to plot (used for filtering and title).
        output_path: Path where the plot will be saved (SVG).
        figsize: Figure size in inches (width, height).
    """
    output_path = Path(output_path)

    # Get unique (dataset, subset) combinations
    # Sort by dataset, then put "global" first within each dataset
    combinations = metrics_df[["dataset", "subset"]].drop_duplicates()
    combinations["sort_key"] = combinations["subset"].apply(
        lambda x: (0, x) if x == "global" else (1, x)
    )
    combinations = combinations.sort_values(["dataset", "sort_key"]).drop(
        columns=["sort_key"]
    )

    # Calculate grid dimensions
    n_plots = len(combinations)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (_, row) in enumerate(combinations.iterrows()):
        dataset = row["dataset"]
        subset = row["subset"]

        # Filter data for this combination (use first metric for this dataset)
        dataset_data = metrics_df[metrics_df["dataset"] == dataset]
        primary_metric = dataset_data["metric"].iloc[0]

        plot_data = metrics_df[
            (metrics_df["dataset"] == dataset)
            & (metrics_df["subset"] == subset)
            & (metrics_df["metric"] == primary_metric)
        ]

        # Plot each score type
        ax = axes[idx]
        for score_type in sorted(plot_data["score_type"].unique()):
            score_data = plot_data[plot_data["score_type"] == score_type]
            ax.plot(
                score_data["step"],
                score_data["value"],
                marker="o",
                label=score_type,
                linewidth=2,
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{primary_metric}")
        ax.set_title(f"{dataset}\n{subset}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Model: {model_name}", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _draw_baseline_lines(
    ax: plt.Axes,
    baselines: dict[str, float],
) -> None:
    """Draw horizontal dashed lines with text labels for baseline model performance.

    Args:
        ax: Matplotlib axes to draw on.
        baselines: Dictionary mapping baseline model names to their metric values.
    """
    if not baselines:
        return

    sorted_baselines = sorted(baselines.items(), key=lambda x: x[1])

    x_max = ax.get_xlim()[1]
    x_pos = x_max * 0.98

    min_separation = 0.01
    prev_y = None
    offset_count = 0

    for name, value in sorted_baselines:
        ax.axhline(y=value, linestyle="--", color="gray", alpha=0.7, linewidth=1)

        if prev_y is not None and abs(value - prev_y) < min_separation:
            offset_count += 1
        else:
            offset_count = 0
        prev_y = value

        y_offset = offset_count * min_separation * 0.5
        ax.text(
            x_pos,
            value + y_offset,
            name,
            fontsize=7,
            color="gray",
            va="bottom",
            ha="right",
        )


def plot_models_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    score_type: str | None = None,
    dataset_subset_score_map: dict[tuple[str, str], str] | None = None,
    figsize: tuple[float, float] | None = None,
    models_filter: list[str] | None = None,
    dataset_subsets_filter: list[tuple[str, str]] | None = None,
    baseline_data: dict[tuple[str, str], dict[str, float]] | None = None,
    title: str | None = None,
    subplot_titles: dict[tuple[str, str], str] | None = None,
    n_cols: int | None = None,
    model_labels: dict[str, str] | None = None,
    model_colors: dict[str, str] | None = None,
    ylim: tuple[float, float] | None = None,
    scale: float = 1.0,
) -> None:
    """Plot metric values vs training step comparing models for a specific score type.

    Creates a grid of subplots where each panel shows a (dataset, subset) combination.
    Each panel has separate lines for each model.

    Args:
        metrics_df: DataFrame with columns [step, dataset, metric, score_type, subset, value, model].
        output_path: Path where the plot will be saved (SVG).
        score_type: Score type to plot for all subplots. Mutually exclusive with dataset_subset_score_map.
        dataset_subset_score_map: Mapping of (dataset, subset) -> score_type for per-subplot score types.
            Mutually exclusive with score_type.
        figsize: Figure size in inches (width, height).
        models_filter: If provided, only include these models. None means include all.
        dataset_subsets_filter: If provided, only include these (dataset, subset) pairs.
            None means include all. Format: [(dataset1, subset1), (dataset2, subset2), ...]
        baseline_data: Optional mapping of (dataset, subset) -> {baseline_name: metric_value}
            for drawing horizontal reference lines.
        title: Override the main plot title. If None, uses default based on score_type.
        subplot_titles: Override titles for specific subplots. Maps (dataset, subset) to title string.
        n_cols: Number of columns in the subplot grid. If None, defaults to min(3, n_plots).
        model_labels: Mapping of model name to display label for legend. If None, uses model name.
        model_colors: Mapping of model name to color (hex code or named color). If None, uses
            matplotlib's default color cycle.
        ylim: Optional y-axis limits as (min, max) tuple. If None, uses auto-scaling.
        scale: Scale factor for subplot size. Default 1.0 gives 4 inches per subplot.
            Use >1.0 for larger plots, <1.0 for smaller.
    """
    output_path = Path(output_path)

    # Validate that exactly one of score_type or dataset_subset_score_map is provided
    if score_type is None and dataset_subset_score_map is None:
        msg = "Must provide either score_type or dataset_subset_score_map"
        raise ValueError(msg)
    if score_type is not None and dataset_subset_score_map is not None:
        msg = "Cannot provide both score_type and dataset_subset_score_map"
        raise ValueError(msg)

    # Apply global filters first
    filtered_df = metrics_df.copy()

    if models_filter is not None:
        filtered_df = filtered_df[filtered_df["model"].isin(models_filter)]

    # Handle score_type filtering
    if score_type is not None:
        # Single score type for all subplots
        filtered_df = filtered_df[filtered_df["score_type"] == score_type]
        if filtered_df.empty:
            msg = f"No data found for score_type: {score_type}"
            raise ValueError(msg)

        # Apply dataset_subsets_filter if provided
        if dataset_subsets_filter is not None:
            mask = pd.Series(False, index=filtered_df.index)
            for dataset, subset in dataset_subsets_filter:
                mask |= (filtered_df["dataset"] == dataset) & (
                    filtered_df["subset"] == subset
                )
            filtered_df = filtered_df[mask]

        if filtered_df.empty:
            msg = "No data found after applying filters"
            raise ValueError(msg)

        # Get unique (dataset, subset) combinations
        combinations = filtered_df[["dataset", "subset"]].drop_duplicates()
    else:
        # Different score type per subplot
        # Get combinations from the map
        combinations_list = list(dataset_subset_score_map.keys())
        combinations = pd.DataFrame(combinations_list, columns=["dataset", "subset"])

        # Filter to only include the (dataset, subset) pairs in the map
        mask = pd.Series(False, index=filtered_df.index)
        for dataset, subset in combinations_list:
            mask |= (filtered_df["dataset"] == dataset) & (
                filtered_df["subset"] == subset
            )
        filtered_df = filtered_df[mask]

        if filtered_df.empty:
            msg = "No data found after applying filters"
            raise ValueError(msg)

    # Sort combinations only if not using dataset_subset_score_map (preserve config order)
    if dataset_subset_score_map is None:
        combinations["sort_key"] = combinations["subset"].apply(
            lambda x: (0, x) if x == "global" else (1, x)
        )
        combinations = combinations.sort_values(["dataset", "sort_key"]).drop(
            columns=["sort_key"]
        )

    # Calculate grid dimensions with square subplots
    n_plots = len(combinations)
    if n_cols is None:
        n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    subplot_size = 4 * scale  # inches per subplot

    if figsize is None:
        figsize = (subplot_size * n_cols, subplot_size * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True
    )
    axes_flat = axes.flatten()

    for idx, (_, row) in enumerate(combinations.iterrows()):
        dataset = row["dataset"]
        subset = row["subset"]

        # Determine which score type to use for this subplot
        if dataset_subset_score_map is not None:
            subplot_score_type = dataset_subset_score_map[(dataset, subset)]
        else:
            subplot_score_type = score_type

        # Filter data for this combination
        subplot_data = filtered_df[
            (filtered_df["dataset"] == dataset) & (filtered_df["subset"] == subset)
        ]

        # If using per-subplot score types, filter by the appropriate score type
        if dataset_subset_score_map is not None:
            subplot_data = subplot_data[
                subplot_data["score_type"] == subplot_score_type
            ]

        # Use first metric for this dataset
        if not subplot_data.empty:
            primary_metric = subplot_data["metric"].iloc[0]
            plot_data = subplot_data[subplot_data["metric"] == primary_metric]
        else:
            continue

        # Plot each model (preserve order from models_filter if provided)
        ax = axes_flat[idx]
        available_models = set(plot_data["model"].unique())
        if models_filter is not None:
            ordered_models = [m for m in models_filter if m in available_models]
        else:
            ordered_models = sorted(available_models)
        for model in ordered_models:
            model_data = plot_data[plot_data["model"] == model]
            display_label = (
                model_labels.get(model, model) if model_labels is not None else model
            )
            color = model_colors.get(model) if model_colors is not None else None
            ax.plot(
                model_data["step"],
                model_data["value"],
                marker="o",
                label=display_label,
                linewidth=2,
                color=color,
            )

        if baseline_data is not None:
            key = (dataset, subset)
            if key in baseline_data:
                _draw_baseline_lines(ax, baseline_data[key])

        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{primary_metric}")

        # Build sample size suffix for title if available
        if "n_pos" in plot_data.columns and "n_neg" in plot_data.columns:
            n_pos = plot_data["n_pos"].iloc[0]
            n_neg = plot_data["n_neg"].iloc[0]
            sample_suffix = f"\n(n={n_pos} vs. {n_neg})"
        else:
            sample_suffix = ""

        # Set subplot title
        if subplot_titles is not None and (dataset, subset) in subplot_titles:
            ax.set_title(subplot_titles[(dataset, subset)] + sample_suffix, fontsize=10)
        elif dataset_subset_score_map is not None:
            ax.set_title(
                f"{dataset}\n{subset}\n{subplot_score_type}" + sample_suffix,
                fontsize=10,
            )
        else:
            ax.set_title(f"{dataset}\n{subset}" + sample_suffix, fontsize=10)

        # Despine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Apply y-axis limits if specified
        if ylim is not None:
            ax.set_ylim(ylim)

    # Hide x-axis labels for non-bottom rows (shared x-axis)
    for idx in range(n_plots):
        row = idx // n_cols
        is_bottom_row = (row == n_rows - 1) or (idx + n_cols >= n_plots)
        if not is_bottom_row:
            axes_flat[idx].set_xlabel("")

    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Set main title
    if title is not None:
        plt.suptitle(title, fontsize=14, y=0.995)
    elif score_type is not None:
        plt.suptitle(f"Score Type: {score_type}", fontsize=14, y=0.995)
    else:
        plt.suptitle("Model Comparison", fontsize=14, y=0.995)

    # Add single shared legend if multiple models
    unique_models = filtered_df["model"].unique()
    if len(unique_models) > 1:
        plt.tight_layout()
        # Get handles and labels from the first subplot
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=10,
            frameon=False,
        )
    else:
        plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
