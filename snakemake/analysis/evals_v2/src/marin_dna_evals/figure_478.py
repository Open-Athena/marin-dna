"""Compact decision figure for issue #478."""

from __future__ import annotations

import re
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
    grid.figure.subplots_adjust(top=0.8, wspace=0.12)
    grid.figure.suptitle("Non-repeat loss by conservation")

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
