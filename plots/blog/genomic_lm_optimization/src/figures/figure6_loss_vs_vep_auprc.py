"""Parameter scaling: matched-region validation LL versus downstream AUPRC.

Both zero-shot LLR and the frozen-embedding linear probe use the paired,
chromosome-weighted AUPRC table from Figure 5. Validation loss is matched to
the consequence's training region and negated so better language modeling reads
left-to-right as higher ``LL (-loss)``.

Run from ``plots/blog/genomic_lm_optimization/src``:

    uv run --project ../../../.. python -m figures.figure6_loss_vs_vep_auprc

Outputs:

    plots/output/blog/genomic_lm_optimization/
        figure6_loss_vs_vep_auprc.{png,pdf,svg}
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figures.data import SCALING_RESULTS_PATH, save
from marin_dna.blog_figure_typography import (
    FIGURE_NOTE_SIZE_PX,
)
from figures.figure5_params_vs_vep_auprc import (
    MENDELIAN_SUBSETS,
    READOUTS,
    SGE_SUBSETS,
    load_parameter_scaling_metrics,
)
from utils.figure_style import X_LABEL_PAD, figsize

MATCHED_LOSS_COLUMN = {
    "missense_variant": "eval_loss_cds",
    "splicing": "eval_loss_cds",
    "synonymous_variant": "eval_loss_cds",
    "tss_proximal": "eval_loss_upstream",
    "5_prime_UTR_variant": "eval_loss_upstream",
    "3_prime_UTR_variant": "eval_loss_downstream",
}
LOSS_COLUMNS = tuple(sorted(set(MATCHED_LOSS_COLUMN.values())))


def load_loss_vs_auprc(
    results: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join paired AUPRCs to each consequence's matched validation LL."""
    if results is None:
        results = pd.read_csv(SCALING_RESULTS_PATH)
    if metrics is None:
        metrics = load_parameter_scaling_metrics()

    required_results = {"params", *LOSS_COLUMNS}
    assert required_results.issubset(results.columns), (
        f"scaling results missing {sorted(required_results - set(results.columns))}"
    )
    metadata = results[["params", *LOSS_COLUMNS]].copy()
    assert len(metadata) == 8
    assert metadata["params"].is_unique
    assert np.isfinite(metadata[list(LOSS_COLUMNS)].to_numpy(dtype=float)).all()

    data = metrics.merge(metadata, on="params", how="left", validate="many_to_one")
    assert len(data) == len(metrics)
    data["ll"] = np.nan
    for subset, loss_column in MATCHED_LOSS_COLUMN.items():
        mask = data["subset"] == subset
        data.loc[mask, "ll"] = -data.loc[mask, loss_column]
    assert data["ll"].notna().all()
    assert np.isfinite(data["ll"].to_numpy(dtype=float)).all()
    return data


def _plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    show_ylabel: bool,
) -> None:
    """Draw paired protocol fits and endpoint AUPRCs for one consequence."""
    assert set(data["score_type"]) == {score for score, *_ in READOUTS}
    for index, (score_type, _label, color, marker, linestyle) in enumerate(READOUTS):
        series = data[data["score_type"] == score_type].sort_values("ll")
        assert len(series) == 8
        assert series["ll"].is_unique
        xs = series["ll"].to_numpy(dtype=float)
        ys = series["value"].to_numpy(dtype=float) * 100.0
        ax.errorbar(
            xs,
            ys,
            yerr=series["se"] * 100.0,
            color=color,
            ecolor=color,
            marker=marker,
            linestyle="none",
            elinewidth=0.9,
            capsize=0,
            markersize=5,
            markeredgecolor="#1f1e1b",
            markeredgewidth=0.45,
            zorder=3,
        )
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.array([xs.min(), xs.max()])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            zorder=2,
        )
        correlation = float(np.corrcoef(xs, ys)[0, 1])
        assert np.isfinite(correlation)
        ax.text(
            0.04,
            0.96 - index * 0.16,
            rf"$r$ = {correlation:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FIGURE_NOTE_SIZE_PX,
            color=color,
        )

    ax.set_title(title)
    ax.set_xlabel("LL (−loss)", labelpad=X_LABEL_PAD)
    if show_ylabel:
        ax.set_ylabel("AUPRC (%)")
    ax.grid(False)
    ax.margins(x=0.08, y=0.13)
    ax.set_box_aspect(1)


def build(
    results: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> None:
    """Build the combined Mendelian + SGE LL-versus-AUPRC figure."""
    data = load_loss_vs_auprc(results, metrics)

    mosaic = [
        [
            "m_missense",
            "m_missense",
            "m_splicing",
            "m_splicing",
            "m_synonymous",
            "m_synonymous",
        ],
        ["m_promoter", "m_promoter", "m_5utr", "m_5utr", "m_3utr", "m_3utr"],
        [".", "s_missense", "s_missense", "s_splicing", "s_splicing", "."],
    ]
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=figsize(10.0, 9.2),
        gridspec_kw={"hspace": 0.52, "wspace": 0.16},
    )

    mendelian_axes = (
        ("m_missense", "missense_variant", "Missense"),
        ("m_splicing", "splicing", "Splicing"),
        ("m_synonymous", "synonymous_variant", "Synonymous"),
        ("m_promoter", "tss_proximal", "Promoter"),
        ("m_5utr", "5_prime_UTR_variant", "5′ UTR"),
        ("m_3utr", "3_prime_UTR_variant", "3′ UTR"),
    )
    assert tuple((subset, title) for _, subset, title in mendelian_axes) == tuple(
        MENDELIAN_SUBSETS
    )
    for index, (axis_name, subset, title) in enumerate(mendelian_axes):
        panel = data[
            (data["dataset"] == "mendelian_traits") & (data["subset"] == subset)
        ]
        _plot_panel(
            axes[axis_name],
            panel,
            title=title,
            show_ylabel=index in (0, 3),
        )

    sge_axes = (
        ("s_missense", "missense_variant", "Missense"),
        ("s_splicing", "splicing", "Splicing"),
    )
    assert tuple((subset, title) for _, subset, title in sge_axes) == tuple(SGE_SUBSETS)
    for index, (axis_name, subset, title) in enumerate(sge_axes):
        panel = data[(data["dataset"] == "sge") & (data["subset"] == subset)]
        _plot_panel(
            axes[axis_name],
            panel,
            title=title,
            show_ylabel=index == 0,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markeredgecolor="#1f1e1b",
            markeredgewidth=0.45,
            markersize=5,
        )
        for _score, _label, color, marker, linestyle in READOUTS
    ]
    labels = [label for _score, label, *_rest in READOUTS]
    fig.legend(
        handles,
        labels,
        title="Scoring protocol",
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.canvas.draw()
    mendelian_top = max(axes[name].get_position().y1 for name, *_ in mendelian_axes)
    sge_top = max(axes[name].get_position().y1 for name, *_ in sge_axes)
    fig.text(0.02, mendelian_top + 0.012, "Mendelian", weight="bold")
    fig.text(0.02, sge_top + 0.012, "SGE", weight="bold")
    save(fig, "figure6_loss_vs_vep_auprc")


if __name__ == "__main__":
    build()
