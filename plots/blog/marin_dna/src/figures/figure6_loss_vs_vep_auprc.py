"""Parameter scaling: matched-region validation LL versus downstream AUPRC.

Both zero-shot LLR and the linear probe use the paired,
chromosome-weighted AUPRC table from Figure 5. Validation loss is matched to
the consequence's training region and negated so better language modeling reads
left-to-right as higher ``LL (-loss)``.

Run from ``plots/blog/marin_dna/src``:

    uv run --project ../../../.. python -m figures.figure6_loss_vs_vep_auprc

Outputs:

    plots/output/blog/marin_dna/
        figure6_loss_vs_vep_auprc.{png,pdf,svg}
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figures.data import SCALING_RESULTS_PATH, save
from marin_dna.blog_figure_typography import (
    MATPLOTLIB_NOTE_SIZE,
)
from figures.figure5_params_vs_vep_auprc import (
    MENDELIAN_SUBSETS,
    READOUTS,
    REDUNDANT_LABEL_HEIGHT_FONT_SIZES,
    SGE_SUBSETS,
    SUBPLOT_HEIGHT_PX,
    load_parameter_scaling_metrics,
)
from utils.figure_style import (
    COMPARISON_ERRORBAR_ALPHA,
    MODEL_FAMILY_MARKERS,
    SCORING_PROTOCOL_COLORS,
    SCORING_PROTOCOL_LINESTYLES,
    X_LABEL_PAD,
    center_axes_block,
    figsize,
    pack_horizontal_axes,
    pack_horizontal_axis_columns,
    set_square_subplot_height,
)

MATCHED_LOSS_COLUMN = {
    "missense_variant": "eval_loss_cds",
    "splicing": "eval_loss_cds",
    "synonymous_variant": "eval_loss_cds",
    "tss_proximal": "eval_loss_upstream",
    "5_prime_UTR_variant": "eval_loss_upstream",
    "3_prime_UTR_variant": "eval_loss_downstream",
}
LOSS_COLUMNS = tuple(sorted(set(MATCHED_LOSS_COLUMN.values())))
ANNOTATION_HEADROOM_SUBSETS = frozenset(
    {"synonymous_variant", "3_prime_UTR_variant"}
)
ANNOTATION_HEADROOM_FRACTION = 0.3


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
    add_annotation_headroom: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Draw paired protocol fits and endpoint AUPRCs for one consequence."""
    assert set(data["score_type"]) == {score for score, *_ in READOUTS}
    marker = MODEL_FAMILY_MARKERS["marindna"]
    for index, (score_type, _label, protocol) in enumerate(READOUTS):
        series = data[data["score_type"] == score_type].sort_values("ll")
        assert len(series) == 8
        assert series["ll"].is_unique
        color = SCORING_PROTOCOL_COLORS[protocol]
        xs = series["ll"].to_numpy(dtype=float)
        ys = series["value"].to_numpy(dtype=float) * 100.0
        ax.errorbar(
            xs,
            ys,
            yerr=series["se"] * 100.0,
            fmt="none",
            ecolor=color,
            alpha=COMPARISON_ERRORBAR_ALPHA,
            capsize=0,
            zorder=1,
        )
        ax.scatter(
            xs,
            ys,
            color=color,
            marker=marker,
            edgecolors="#1f1e1b",
            zorder=3,
        )
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.array([xs.min(), xs.max()])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=color,
            linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
            zorder=2,
        )
        correlation = float(np.corrcoef(xs, ys)[0, 1])
        assert np.isfinite(correlation)
        ax.text(
            0.04,
            0.96 - index * 0.16,
            f"$r$ =\u2009{correlation:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=MATPLOTLIB_NOTE_SIZE,
            color=color,
        )

    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("LL (−loss)", labelpad=X_LABEL_PAD)
    if show_ylabel:
        ax.set_ylabel("AUPRC (%)")
    ax.grid(False)
    ax.margins(x=0.08)
    if add_annotation_headroom:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * ANNOTATION_HEADROOM_FRACTION)
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
            add_annotation_headroom=subset in ANNOTATION_HEADROOM_SUBSETS,
            show_xlabel=index >= 3,
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
            add_annotation_headroom=False,
            show_xlabel=True,
            show_ylabel=index == 0,
        )

    set_square_subplot_height(fig, axes.values(), SUBPLOT_HEIGHT_PX)
    mendelian_rows = (
        tuple(axes[axis_name] for axis_name, *_ in mendelian_axes[:3]),
        tuple(axes[axis_name] for axis_name, *_ in mendelian_axes[3:]),
    )
    # Loss tick labels are wider than the parameter ticks in Figure 12. Pack
    # their rendered bounds flush so the square axes themselves keep the same
    # column spacing in both figures.
    pack_horizontal_axis_columns(fig, mendelian_rows, gap_font_sizes=0.0)
    sge_row = tuple(axes[axis_name] for axis_name, *_ in sge_axes)
    pack_horizontal_axes(fig, sge_row)
    center_axes_block(fig, sge_row, (axis for row in mendelian_rows for axis in row))
    freed_label_height = (
        plt.rcParams["font.size"]
        * REDUNDANT_LABEL_HEIGHT_FONT_SIZES
        / (fig.get_figheight() * 72.0)
    )
    for axis_name, *_ in (*mendelian_axes[3:], *sge_axes):
        position = axes[axis_name].get_position()
        axes[axis_name].set_position(
            (
                position.x0,
                position.y0 + freed_label_height,
                position.width,
                position.height,
            )
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=SCORING_PROTOCOL_COLORS[protocol],
            linestyle=SCORING_PROTOCOL_LINESTYLES[protocol],
        )
        for _score, _label, protocol in READOUTS
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
