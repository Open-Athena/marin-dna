"""Figure 6: corresponding-region validation loss vs VEP AUPRC by variant type.

Redo of the blog's Fig 6 (Eric's 2×3 scatter) on the **new** Mendelian eval:
x = final validation loss for the training-data region corresponding to the
variant type (vendored scaling CSV), y = new-eval AUPRC (``blog_metrics``),
marker color = params. Dotted linear best-fit + Pearson r per panel make the
blog's point — lower loss does **not** cleanly predict better VEP.

Run:  uv run python -m plots.blog.figure6_loss_vs_vep_auprc
Out:  plots/output/blog/figure6_loss_vs_vep_auprc__mendelian_llr.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.blog._regions import MENDELIAN_VARIANT_ORDER, REGION_LABELS
from plots.blog._scaling import VEP_PANELS, ladder_llr_table, ladder_probe_table
from plots.blog._style.figure_style import (
    FIGURE_WIDTH,
    X_LABEL_PAD,
    attach_params_legend_below,
    figsize,
    palette,
)
from plots.blog._style.savefig import save_figure

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"
_MARKER_AREA = 110.0


def plot_loss_panel(ax, data, param_palette) -> None:
    """Draw one params-colored loss→AUPRC panel with a Pearson-r annotation."""
    assert len(data) >= 2
    assert data["loss_region"].nunique() == 1
    xs = data["eval_loss"].to_numpy(dtype=float)
    ys = data["value"].to_numpy(dtype=float)
    assert np.isfinite(xs).all() and np.isfinite(ys).all()

    for _, row in data.iterrows():
        ax.scatter(
            row["eval_loss"],
            row["value"],
            s=_MARKER_AREA,
            color=param_palette[int(row["params"])],
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )
    slope, intercept = np.polyfit(xs, ys, 1)
    x_line = np.array([xs.min(), xs.max()])
    ax.plot(
        x_line,
        slope * x_line + intercept,
        linestyle=":",
        color="0.35",
        linewidth=1.2,
        zorder=2,
    )
    pearson_r = float(np.corrcoef(xs, ys)[0, 1])
    ax.text(
        0.97,
        0.95,
        rf"$r = {pearson_r:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
    )
    region = str(data["loss_region"].iloc[0])
    ax.set_xlabel(f"{REGION_LABELS[region]} validation loss", labelpad=X_LABEL_PAD)
    ax.grid(False)
    ax.margins(y=0.12)


def build(world: str, data, metric_label: str) -> None:
    params_present = sorted({int(p) for p in data["params"].unique()})
    pal = palette(params_present)
    title_for = {subset: title for subset, title in VEP_PANELS}

    fig, axes = plt.subplots(2, 3, figsize=figsize(FIGURE_WIDTH, 7.0))
    for ax, subset in zip(axes.flat, MENDELIAN_VARIANT_ORDER, strict=True):
        d = data[data["subset"] == subset].sort_values("params")
        n = int(d["n"].iloc[0]) if len(d) else 0
        plot_loss_panel(ax, d, pal)
        title = title_for[subset]
        ax.set_title(f"{title[:1].upper() + title[1:]} (n={n:,})", fontsize=10)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUPRC")

    fig.suptitle(
        "Parameter scaling — corresponding-region validation loss vs VEP AUPRC "
        f"· {metric_label}",
        fontsize=11,
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    attach_params_legend_below(
        fig, pal, params_present, width_scale=0.4, handlelength=1.0
    )
    save_figure(fig, OUTPUT_DIR, f"figure6_loss_vs_vep_auprc__mendelian_{world}")


def build_all() -> None:
    build("llr", ladder_llr_table(), "zero-shot LLR (new eval)")
    build("probe", ladder_probe_table(), "frozen-embedding linear probe")


if __name__ == "__main__":
    build_all()
