"""Single-orientation sensitivity figure for issue #478."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marin_dna_evals.figure_478 import REGION_COLORS, _model_label

ORIENTATION_COLORS = {
    "fwd": "#CC79A7",
    "rc": "#E69F00",
    "fwd_rc_mean": "#222222",
}
ORIENTATION_LABELS = {
    "fwd": "FWD only",
    "rc": "RC only",
    "fwd_rc_mean": "FWD/RC mean",
}


def _plot_orientation_curves(
    ax,
    primary: pd.DataFrame,
    *,
    orientation: str,
    show_legend: bool,
) -> None:
    scale = primary[
        (primary["orientation"] == orientation)
        & (primary["score_kind"] == "absolute_nll")
    ]
    model_order = scale["model_from"].drop_duplicates().tolist()
    assert model_order
    x = np.arange(len(model_order))
    for region in ("cds", "upstream", "downstream"):
        for conserved, linestyle in ((False, "--"), (True, "-")):
            for repeat, marker in ((False, "o"), (True, "s")):
                subset = scale[
                    (scale["region"] == region)
                    & (scale["conserved"] == conserved)
                    & (scale["repeat"] == repeat)
                ].set_index("model_from")
                assert set(model_order) <= set(subset.index)
                subset = subset.loc[model_order]
                ax.plot(
                    x,
                    subset["mean"],
                    marker=marker,
                    markersize=3,
                    linewidth=1.4,
                    linestyle=linestyle,
                    color=REGION_COLORS[region],
                    label=f"{region}; C{int(conserved)} R{int(repeat)}",
                )
    ax.set_xticks(x, [_model_label(model) for model in model_order])
    ax.set_ylabel(f"{ORIENTATION_LABELS[orientation]} NLL (nats/base)")
    if show_legend:
        ax.legend(fontsize=6, ncol=3, frameon=False)


def plot_orientation_sensitivity_478(
    orientation_summary_path: str | Path,
    orientation_controlled_path: str | Path,
    agreement_path: str | Path,
    averaged_controlled_path: str | Path,
    output_path: str | Path,
) -> None:
    """Render all-rung patterns, controlled effects, and per-base agreement."""
    summary = pd.read_parquet(orientation_summary_path)
    orientation_controlled = pd.read_parquet(orientation_controlled_path)
    agreement = pd.read_parquet(agreement_path)
    averaged_controlled = pd.read_parquet(averaged_controlled_path).assign(
        orientation="fwd_rc_mean"
    )
    primary = summary[
        (summary["analysis_family"] == "primary")
        & (summary["span"] == "central_32_222")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_fwd, ax_rc, ax_controlled, ax_agreement = axes.flat

    _plot_orientation_curves(
        ax_fwd,
        primary,
        orientation="fwd",
        show_legend=True,
    )
    ax_fwd.set_title("A  FWD-only scaling curves")
    _plot_orientation_curves(
        ax_rc,
        primary,
        orientation="rc",
        show_legend=False,
    )
    ax_rc.set_title("B  RC-only scaling curves")

    terms = ("conserved", "repeat", "conserved_x_repeat")
    combined = pd.concat(
        [orientation_controlled, averaged_controlled],
        ignore_index=True,
    )
    endpoint = combined[
        (combined["score_kind"] == "endpoint_delta") & combined["term"].isin(terms)
    ]
    groups = [
        (region, term) for region in ("cds", "upstream", "downstream") for term in terms
    ]
    x = np.arange(len(groups))
    offsets = {"fwd": -0.22, "rc": 0.0, "fwd_rc_mean": 0.22}
    for orientation in ("fwd", "rc", "fwd_rc_mean"):
        rows = []
        for region, term in groups:
            hit = endpoint[
                (endpoint["orientation"] == orientation)
                & (endpoint["region"] == region)
                & (endpoint["term"] == term)
            ]
            assert len(hit) == 1
            rows.append(hit.iloc[0])
        estimates = np.asarray([row["estimate"] for row in rows])
        lows = np.asarray([row["ci_low"] for row in rows])
        highs = np.asarray([row["ci_high"] for row in rows])
        xpos = x + offsets[orientation]
        ax_controlled.errorbar(
            xpos,
            estimates,
            yerr=[estimates - lows, highs - estimates],
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=1,
            color=ORIENTATION_COLORS[orientation],
            label=ORIENTATION_LABELS[orientation],
        )
    term_labels = {
        "conserved": "cons.",
        "repeat": "repeat",
        "conserved_x_repeat": "inter.",
    }
    ax_controlled.axhline(0, color="#333333", linewidth=0.8)
    ax_controlled.set_xticks(
        x,
        [f"{region}\n{term_labels[term]}" for region, term in groups],
        rotation=35,
        ha="right",
    )
    ax_controlled.set_ylabel("Adjusted endpoint coefficient")
    ax_controlled.set_title("C  Controlled endpoint effects")
    ax_controlled.legend(frameon=False, fontsize=8)

    overall = agreement[
        (agreement["score_kind"] == "endpoint_delta")
        & (agreement["conservation"] == "all")
        & (agreement["repeat_status"] == "all")
        & agreement["comparison"].isin(["fwd_vs_mean", "rc_vs_mean"])
    ].set_index(["comparison", "region"])
    regions = ["cds", "upstream", "downstream"]
    metrics = (
        ("spearman_sample", "Spearman"),
        ("top_fraction_overlap", "top-10% overlap"),
        ("sign_agreement", "gain-sign agreement"),
    )
    comparisons = (("fwd_vs_mean", "FWD"), ("rc_vs_mean", "RC"))
    xlabels = [f"{region}\n{label}" for region in regions for _, label in metrics]
    x = np.arange(len(xlabels))
    width = 0.36
    for comparison_index, (comparison, label) in enumerate(comparisons):
        values = [
            overall.loc[(comparison, region), column]
            for region in regions
            for column, _ in metrics
        ]
        ax_agreement.bar(
            x + (comparison_index - 0.5) * width,
            values,
            width=width,
            color=ORIENTATION_COLORS[label.lower()],
            label=f"{label} vs mean",
        )
    ax_agreement.axhline(0.1, color="#777777", linestyle=":", linewidth=0.8)
    ax_agreement.set_xticks(x, xlabels, rotation=35, ha="right")
    ax_agreement.set_ylim(0, 1)
    ax_agreement.set_ylabel("Single orientation versus mean")
    ax_agreement.set_title("D  Half-compute endpoint substitution")
    ax_agreement.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Single-orientation sensitivity: half-inference-compute alternatives",
        fontsize=14,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
