"""Compact decision figure for issue #478."""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REGION_COLORS = {
    "cds": "#0072B2",
    "upstream": "#D55E00",
    "downstream": "#009E73",
}


def _model_label(model: str) -> str:
    match = re.search(r"-p(46M|76M|128M|255M|476M|1B|2B|4B)-", model)
    return match.group(1) if match else model


def _model_parameters(model: str) -> int:
    label = _model_label(model)
    multiplier = 1_000_000_000 if label.endswith("B") else 1_000_000
    return int(label[:-1]) * multiplier


def plot_nonrepeat_conservation_loss_478(
    summary_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot mean absolute loss for conserved and non-conserved non-repeat bases."""
    summary = pd.read_parquet(summary_path)
    data = summary[
        (summary["analysis_family"] == "primary")
        & (summary["span"] == "central_32_222")
        & (summary["score_kind"] == "absolute_nll")
        & (~summary["repeat"])
    ].copy()
    assert len(data) == 48, f"expected 48 non-repeat rows, found {len(data)}"

    data["Parameters"] = data["model_from"].map(_model_parameters)
    data["Loss"] = data["mean"]
    data["Conservation"] = data["conserved"].map(
        {True: "Conserved", False: "Non-conserved"}
    )
    data["Conservation"] = pd.Categorical(
        data["Conservation"],
        categories=["Conserved", "Non-conserved"],
        ordered=True,
    )
    data["Region"] = data["region"].map(
        {"cds": "CDS", "upstream": "Upstream", "downstream": "Downstream"}
    )
    data["Region"] = pd.Categorical(
        data["Region"],
        categories=["CDS", "Upstream", "Downstream"],
        ordered=True,
    )

    sns.set_theme()
    grid = sns.relplot(
        data=data,
        x="Parameters",
        y="Loss",
        hue="Conservation",
        hue_order=["Conserved", "Non-conserved"],
        col="Region",
        col_order=["CDS", "Upstream", "Downstream"],
        kind="line",
        estimator=None,
        errorbar=None,
        marker="o",
        height=3,
        aspect=1,
        facet_kws={"sharey": False},
    )
    grid.set(xscale="log")
    grid.set_axis_labels("Parameters", "Loss")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.set_box_aspect(1)
    grid.figure.subplots_adjust(top=0.8, wspace=0.05)
    grid.figure.suptitle("Loss by conservation (repeats excluded)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def plot_token_composition_478(
    summary_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot conservation by repeat percentages for each region and globally."""
    summary = pd.read_parquet(summary_path)
    data = summary[
        (summary["analysis_family"] == "primary")
        & (summary["span"] == "central_32_222")
        & (summary["score_kind"] == "absolute_nll")
    ].copy()
    cell_columns = ["region", "repeat", "conserved"]
    count_variation = data.groupby(cell_columns, observed=True)["n_positions"].nunique()
    assert (count_variation == 1).all(), "position counts vary across models"
    counts = data.drop_duplicates(cell_columns)[[*cell_columns, "n_positions"]]
    assert len(counts) == 12, f"expected 12 composition cells, found {len(counts)}"

    panels: list[tuple[str, pd.DataFrame]] = [
        (
            "Global",
            counts.groupby(["repeat", "conserved"], as_index=False, observed=True)[
                "n_positions"
            ].sum(),
        ),
        ("CDS", counts[counts["region"] == "cds"]),
        ("Upstream", counts[counts["region"] == "upstream"]),
        ("Downstream", counts[counts["region"] == "downstream"]),
    ]
    matrices: list[tuple[str, pd.DataFrame]] = []
    for title, panel in panels:
        matrix = (
            panel.pivot(index="repeat", columns="conserved", values="n_positions")
            .reindex(index=[False, True], columns=[True, False])
            .astype(float)
        )
        matrix = 100 * matrix / matrix.to_numpy().sum()
        matrix.index = ["Non-repeat", "Repeat"]
        matrix.columns = ["Conserved", "Non-conserved"]
        matrices.append((title, matrix))

    scale_max = max(matrix.to_numpy().max() for _, matrix in matrices)
    sns.set_theme()
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8, 6),
        layout="constrained",
    )
    for index, (axis, (title, matrix)) in enumerate(
        zip(axes.flat, matrices, strict=True)
    ):
        annotation = matrix.map(lambda value: f"{value:.1f}%")
        sns.heatmap(
            matrix,
            annot=annotation,
            fmt="",
            cbar=False,
            square=True,
            vmin=0,
            vmax=scale_max,
            ax=axis,
        )
        axis.set_title(title)
        row, column = divmod(index, 2)
        axis.set_xlabel("Conservation" if row == 1 else "")
        axis.set_ylabel("Repeat status" if column == 0 else "")
        axis.tick_params(axis="both", labelrotation=0)
    figure.colorbar(axes.flat[0].collections[0], ax=axes, label="Tokens (%)")
    figure.suptitle("Conservation and repeat composition")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_conservation_classification_478(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    orientation: str = "fwd_rc_mean",
) -> None:
    """Plot pooled AUPRC for absolute loss and entropy across model sizes."""
    metrics = pd.read_parquet(metrics_path)
    data = metrics[
        (metrics["orientation"] == orientation)
        & metrics["statistic"].isin(["loss", "entropy"])
        & (metrics["model_from"] == metrics["model_to"])
    ].copy()
    model_order = sorted(data["model_from"].unique(), key=_model_parameters)
    scopes = {
        "global": "Global",
        "cds": "CDS",
        "upstream": "Upstream",
        "downstream": "Downstream",
    }
    expected_rows = len(model_order) * 2 * len(scopes)
    assert len(data) == expected_rows, (
        f"expected {expected_rows} absolute-score rows, found {len(data)}"
    )
    assert data[["auprc", "prevalence"]].notna().all().all()

    data["Parameters"] = data["model_from"].map(_model_parameters)
    data["AUPRC"] = data["auprc"]
    data["Statistic"] = data["statistic"].map({"loss": "Loss", "entropy": "Entropy"})
    data["Scope"] = data["scope"].map(scopes)
    data["Scope"] = pd.Categorical(
        data["Scope"],
        categories=list(scopes.values()),
        ordered=True,
    )

    sns.set_theme()
    grid = sns.relplot(
        data=data,
        x="Parameters",
        y="AUPRC",
        hue="Statistic",
        hue_order=["Loss", "Entropy"],
        col="Scope",
        col_order=list(scopes.values()),
        col_wrap=2,
        kind="line",
        estimator=None,
        errorbar=None,
        marker="o",
        height=3,
        aspect=1,
        facet_kws={"sharey": False},
    )
    grid.set(xscale="log")
    grid.set_axis_labels("Parameters", "AUPRC")
    grid.set_titles("{col_name}")
    for scope, axis in zip(scopes, grid.axes.flat, strict=True):
        prevalence = data.loc[data["scope"] == scope, "prevalence"].unique()
        assert len(prevalence) == 1
        axis.axhline(prevalence[0], color="0.5", linestyle="--")
        axis.set_box_aspect(1)
    grid.figure.subplots_adjust(top=0.9, hspace=0.25, wspace=0.08)
    grid.figure.suptitle("Non-repeat conservation classification")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def plot_classification_orientation_478(
    averaged_metrics_path: str | Path,
    orientation_metrics_path: str | Path,
    output_path: str | Path,
    *,
    statistic: str,
) -> None:
    """Compare pooled absolute-score AUPRC with one or two orientations."""
    assert statistic in {"loss", "entropy"}
    metrics = pd.concat(
        [
            pd.read_parquet(averaged_metrics_path),
            pd.read_parquet(orientation_metrics_path),
        ],
        ignore_index=True,
    )
    data = metrics[
        (metrics["statistic"] == statistic)
        & (metrics["model_from"] == metrics["model_to"])
    ].copy()
    model_order = sorted(data["model_from"].unique(), key=_model_parameters)
    scopes = {
        "global": "Global",
        "cds": "CDS",
        "upstream": "Upstream",
        "downstream": "Downstream",
    }
    orientations = {
        "fwd_rc_mean": "FWD/RC mean",
        "fwd": "FWD",
        "rc": "RC",
    }
    expected_rows = len(model_order) * len(scopes) * len(orientations)
    assert len(data) == expected_rows, (
        f"expected {expected_rows} orientation rows, found {len(data)}"
    )

    data["Parameters"] = data["model_from"].map(_model_parameters)
    data["AUPRC"] = data["auprc"]
    data["Scope"] = data["scope"].map(scopes)
    data["Scope"] = pd.Categorical(
        data["Scope"],
        categories=list(scopes.values()),
        ordered=True,
    )
    data["Orientation"] = data["orientation"].map(orientations)

    sns.set_theme()
    grid = sns.relplot(
        data=data,
        x="Parameters",
        y="AUPRC",
        hue="Orientation",
        hue_order=list(orientations.values()),
        col="Scope",
        col_order=list(scopes.values()),
        col_wrap=2,
        kind="line",
        estimator=None,
        errorbar=None,
        marker="o",
        height=3,
        aspect=1,
        facet_kws={"sharey": False},
    )
    grid.set(xscale="log")
    grid.set_axis_labels("Parameters", "AUPRC")
    grid.set_titles("{col_name}")
    for scope, axis in zip(scopes, grid.axes.flat, strict=True):
        prevalence = data.loc[data["scope"] == scope, "prevalence"].unique()
        assert len(prevalence) == 1
        axis.axhline(prevalence[0], color="0.5", linestyle="--")
        axis.set_box_aspect(1)
    grid.figure.subplots_adjust(top=0.9, hspace=0.25, wspace=0.08)
    grid.figure.suptitle(f"Non-repeat {statistic} classification by orientation")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def plot_practical_delta_orientation_478(
    averaged_metrics_path: str | Path,
    orientation_metrics_path: str | Path,
    output_path: str | Path,
) -> None:
    """Compare 46M-to-76M loss-delta AUPRC lift by orientation."""
    metrics = pd.concat(
        [
            pd.read_parquet(averaged_metrics_path),
            pd.read_parquet(orientation_metrics_path),
        ],
        ignore_index=True,
    )
    data = metrics[
        (metrics["statistic"] == "loss_delta")
        & (metrics["model_from"].map(_model_label) == "46M")
        & (metrics["model_to"].map(_model_label) == "76M")
    ].copy()
    scopes = {
        "global": "Global",
        "cds": "CDS",
        "upstream": "Upstream",
        "downstream": "Downstream",
    }
    orientations = {
        "fwd_rc_mean": "FWD/RC mean",
        "fwd": "FWD",
        "rc": "RC",
    }
    expected_rows = len(scopes) * len(orientations)
    assert len(data) == expected_rows, (
        f"expected {expected_rows} practical-delta rows, found {len(data)}"
    )
    data["Scope"] = data["scope"].map(scopes)
    data["Scope"] = pd.Categorical(
        data["Scope"],
        categories=list(scopes.values()),
        ordered=True,
    )
    data["Orientation"] = data["orientation"].map(orientations)
    data["AUPRC − prevalence"] = data["auprc_minus_prevalence"]

    sns.set_theme()
    grid = sns.catplot(
        data=data,
        x="Scope",
        y="AUPRC − prevalence",
        hue="Orientation",
        hue_order=list(orientations.values()),
        kind="bar",
        errorbar=None,
        height=6,
        aspect=1.25,
    )
    grid.set_axis_labels("Region", "AUPRC − prevalence")
    grid.ax.axhline(0, color="0.5", linestyle="--")
    grid.ax.set_box_aspect(1)
    grid.figure.subplots_adjust(top=0.88)
    grid.figure.suptitle("46M→76M loss-delta classification by orientation")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def plot_loss_delta_classification_478(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    orientation: str = "fwd_rc_mean",
) -> None:
    """Plot AUPRC lift over prevalence for every smaller-to-larger loss delta."""
    metrics = pd.read_parquet(metrics_path)
    data = metrics[
        (metrics["orientation"] == orientation) & (metrics["statistic"] == "loss_delta")
    ].copy()
    scopes = {
        "global": "Global",
        "cds": "CDS",
        "upstream": "Upstream",
        "downstream": "Downstream",
    }
    model_order = sorted(
        set(data["model_from"]) | set(data["model_to"]),
        key=_model_parameters,
    )
    expected_rows = len(scopes) * len(list(combinations(model_order, 2)))
    assert len(data) == expected_rows, (
        f"expected {expected_rows} loss-delta rows, found {len(data)}"
    )

    matrices: list[tuple[str, pd.DataFrame]] = []
    for scope, title in scopes.items():
        panel = data[data["scope"] == scope]
        matrix = panel.pivot(
            index="model_to",
            columns="model_from",
            values="auprc_minus_prevalence",
        ).reindex(index=model_order[::-1], columns=model_order)
        matrix.index = [_model_label(model) for model in matrix.index]
        matrix.columns = [_model_label(model) for model in matrix.columns]
        matrices.append((title, matrix))

    matrices = [(title, 100 * matrix) for title, matrix in matrices]
    finite_values = np.concatenate(
        [matrix.to_numpy(dtype=float).ravel() for _, matrix in matrices]
    )
    scale_limit = float(np.nanmax(np.abs(finite_values)))
    sns.set_theme()
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(9, 8),
        layout="constrained",
    )
    for index, (axis, (title, matrix)) in enumerate(
        zip(axes.flat, matrices, strict=True)
    ):
        annotation = matrix.map(lambda value: f"{value:.0f}")
        axis.set_facecolor("white")
        annotation = annotation.mask(annotation == "-0", "0")
        sns.heatmap(
            matrix,
            annot=annotation,
            fmt="",
            mask=matrix.isna(),
            cbar=False,
            cmap="vlag",
            center=0,
            square=True,
            vmin=-scale_limit,
            vmax=scale_limit,
            ax=axis,
        )
        axis.set_title(title)
        row, column = divmod(index, 2)
        axis.set_xlabel("Smaller model" if row == 1 else "")
        axis.set_ylabel("Larger model" if column == 0 else "")
        axis.tick_params(axis="x", labelrotation=45)
        axis.tick_params(axis="y", labelrotation=0)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    figure.colorbar(
        axes.flat[0].collections[0],
        ax=axes,
        label="AUPRC lift (%)",
    )
    figure.suptitle("Loss-delta conservation classification")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_compute_efficiency_478(
    orientation_metrics_path: str | Path,
    output_path: str | Path,
) -> None:
    """Compare global conservation AUPRC against approximate scoring compute."""
    metrics = pd.read_parquet(orientation_metrics_path)
    data = metrics[
        (metrics["scope"] == "global") & (metrics["orientation"] == "fwd")
    ].copy()
    model_order = sorted(
        set(data["model_from"]) | set(data["model_to"]),
        key=_model_parameters,
    )
    scores_per_orientation = 2 * len(model_order) + len(
        list(combinations(model_order, 2))
    )
    expected_rows = scores_per_orientation
    assert len(data) == expected_rows, (
        f"expected {expected_rows} global compute-comparison rows, found {len(data)}"
    )

    parameter_passes = data["model_from"].map(_model_parameters).astype(float)
    is_delta = data["statistic"] == "loss_delta"
    parameter_passes.loc[is_delta] += data.loc[is_delta, "model_to"].map(
        _model_parameters
    )
    data["Relative scoring compute"] = parameter_passes / 46_000_000
    data["AUPRC (%)"] = 100 * data["auprc"]
    data["Approach"] = data["statistic"].map(
        {"loss": "Loss", "entropy": "Entropy", "loss_delta": "Loss delta"}
    )

    sns.set_theme()
    grid = sns.relplot(
        data=data,
        x="Relative scoring compute",
        y="AUPRC (%)",
        hue="Approach",
        hue_order=["Loss", "Entropy", "Loss delta"],
        kind="scatter",
        height=6,
        aspect=1,
    )
    grid.set(xscale="log")
    grid.set_axis_labels("Relative scoring compute", "AUPRC (%)")
    prevalence = data["prevalence"].unique()
    assert len(prevalence) == 1
    grid.ax.axhline(100 * prevalence[0], color="0.5", linestyle="--")
    grid.ax.set_box_aspect(1)
    grid.figure.subplots_adjust(top=0.9)
    grid.figure.suptitle("Global AUPRC by relative FWD scoring compute")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def _errorbar(ax, x, row: pd.Series, *, color: str) -> None:
    ax.errorbar(
        x,
        row["mean"],
        yerr=[
            [row["mean"] - row["ci_low"]],
            [row["ci_high"] - row["mean"]],
        ],
        fmt="none",
        ecolor=color,
        elinewidth=1,
        capsize=2,
        alpha=0.8,
    )


def plot_predictability_478(
    summary_path: str | Path,
    controlled_path: str | Path,
    output_path: str | Path,
) -> None:
    """Render the primary scaling/endpoint result and CDS-only diagnostics."""
    summary = pd.read_parquet(summary_path)
    controlled = pd.read_parquet(controlled_path)
    primary = summary[
        (summary["analysis_family"] == "primary")
        & (summary["span"] == "central_32_222")
    ]
    model_order = (
        primary[primary["score_kind"] == "absolute_nll"]["model_from"]
        .drop_duplicates()
        .tolist()
    )
    assert model_order, "no absolute-NLL rows"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_scale, ax_endpoint, ax_control, ax_secondary = axes.flat

    # A: all primary 2x2 strata across scale. Region is color, conservation is
    # line style, and repeat status is marker shape.
    scale = primary[primary["score_kind"] == "absolute_nll"]
    x = np.arange(len(model_order))
    for region in ("cds", "upstream", "downstream"):
        for conserved, linestyle in ((False, "--"), (True, "-")):
            for repeat, marker in ((False, "o"), (True, "s")):
                subset = scale[
                    (scale["region"] == region)
                    & (scale["conserved"] == conserved)
                    & (scale["repeat"] == repeat)
                ].set_index("model_from")
                if not set(model_order) <= set(subset.index):
                    continue
                subset = subset.loc[model_order]
                label = f"{region}; C{int(conserved)} R{int(repeat)}"
                ax_scale.plot(
                    x,
                    subset["mean"],
                    marker=marker,
                    markersize=3,
                    linewidth=1.4,
                    linestyle=linestyle,
                    color=REGION_COLORS[region],
                    label=label,
                )
                ax_scale.fill_between(
                    x,
                    subset["ci_low"].to_numpy(dtype=float),
                    subset["ci_high"].to_numpy(dtype=float),
                    color=REGION_COLORS[region],
                    alpha=0.04,
                )
    ax_scale.set_xticks(x, [_model_label(model) for model in model_order])
    ax_scale.set_ylabel("FWD/RC-averaged NLL (nats/base)")
    ax_scale.set_title("A  Absolute predictability by primary stratum")
    ax_scale.legend(fontsize=6, ncol=3, frameon=False)

    # B: unadjusted endpoint improvement in all primary 2×2 strata.
    endpoint = primary[primary["score_kind"] == "endpoint_delta"].copy()
    labels, rows, colors = [], [], []
    for region in ("cds", "upstream", "downstream"):
        for conserved, repeat in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            hit = endpoint[
                (endpoint["region"] == region)
                & (endpoint["conserved"] == conserved)
                & (endpoint["repeat"] == repeat)
            ]
            if len(hit) != 1:
                continue
            labels.append(f"{region}\nC{int(conserved)} R{int(repeat)}")
            rows.append(hit.iloc[0])
            colors.append(REGION_COLORS[region])
    for index, (row, color) in enumerate(zip(rows, colors)):
        ax_endpoint.bar(index, row["mean"], color=color, alpha=0.82)
        _errorbar(ax_endpoint, index, row, color="#222222")
    ax_endpoint.axhline(0, color="#333333", linewidth=0.8)
    ax_endpoint.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax_endpoint.set_ylabel("NLL(46M) − NLL(4B)")
    ax_endpoint.set_title("B  Endpoint gain by primary stratum")

    # C: adjusted endpoint coefficients, with the interaction directly exposing
    # whether conserved-repeat overlap behaves non-additively.
    terms = ["conserved", "repeat", "conserved_x_repeat"]
    controlled_endpoint = controlled[
        (controlled["score_kind"] == "endpoint_delta") & controlled["term"].isin(terms)
    ]
    width = 0.24
    for region_index, region in enumerate(("cds", "upstream", "downstream")):
        subset = controlled_endpoint[controlled_endpoint["region"] == region].set_index(
            "term"
        )
        if not set(terms) <= set(subset.index):
            continue
        subset = subset.loc[terms]
        xpos = np.arange(len(terms)) + (region_index - 1) * width
        ax_control.bar(
            xpos,
            subset["estimate"],
            width=width,
            color=REGION_COLORS[region],
            label=region,
        )
        ax_control.errorbar(
            xpos,
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["ci_low"],
                subset["ci_high"] - subset["estimate"],
            ],
            fmt="none",
            ecolor="#222222",
            capsize=2,
            linewidth=1,
        )
    ax_control.axhline(0, color="#333333", linewidth=0.8)
    ax_control.set_xticks(
        range(len(terms)),
        ["conserved", "repeat", "interaction"],
    )
    ax_control.set_ylabel("Adjusted endpoint coefficient")
    ax_control.set_title("C  GC / position / 7-mer adjusted")
    ax_control.legend(frameon=False, fontsize=8)

    # D: CDS-only secondary features. Pool the primary conservation/repeat cells
    # by their position counts; these are diagnostic, not the headline contrast.
    secondary = summary[
        summary["analysis_family"].isin(["secondary_codon", "secondary_splice"])
        & (summary["score_kind"] == "endpoint_delta")
    ].copy()
    if len(secondary):
        secondary["weighted"] = secondary["mean"] * secondary["n_positions"]
        pooled = secondary.groupby(
            ["analysis_family", "feature", "feature_strand"],
            as_index=False,
        ).agg(weighted=("weighted", "sum"), n=("n_positions", "sum"))
        pooled["mean"] = pooled["weighted"] / pooled["n"]
        display = {
            "codon_1": "codon 1",
            "codon_2": "codon 2",
            "codon_3": "codon 3",
            "splice_donor_2bp": "donor 2 bp",
            "splice_acceptor_2bp": "acceptor 2 bp",
        }
        pooled["label"] = (
            pooled["feature"].map(display) + " " + pooled["feature_strand"].str[0]
        )
        pooled = pooled.sort_values(["analysis_family", "feature", "feature_strand"])
        ax_secondary.barh(
            np.arange(len(pooled)),
            pooled["mean"],
            color="#56B4E9",
        )
        ax_secondary.set_yticks(np.arange(len(pooled)), pooled["label"])
    ax_secondary.axvline(0, color="#333333", linewidth=0.8)
    ax_secondary.set_xlabel("NLL(46M) − NLL(4B)")
    ax_secondary.set_title("D  CDS-only secondary diagnostics")

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Conservation × repeat predictability across the scaling ladder",
        fontsize=14,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
