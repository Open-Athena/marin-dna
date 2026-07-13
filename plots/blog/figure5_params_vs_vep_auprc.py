"""Figure 5 (M·LLR): parameter scaling — params vs VEP AUPRC by variant type.

Redo of the blog's Fig 5 in Eric's 1×3 style, but AUPRC comes from the **new**
Mendelian offline eval — via ``blog_metrics.read_llr_metrics`` on the 8 scaling-
ladder endpoints (``…-step-215573``) — instead of the old wandb in-training
numbers. ``params`` come from the vendored ``parameter_scaling_results.csv``
(training results are eval-independent).

Run:  uv run python -m plots.blog.figure5_params_vs_vep_auprc
Out:  plots/output/blog/figure5_params_vs_vep_auprc__mendelian_llr.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from plots.blog._regions import (
    MENDELIAN_VARIANT_ORDER,
    REGION_COLORS,
    VARIANT_REGION,
    region_legend_handles,
)
from plots.blog._scaling import VEP_PANELS, ladder_llr_table, ladder_probe_table
from plots.blog._style.figure_style import FIGURE_WIDTH, X_LABEL_PAD, figsize
from plots.blog._style.savefig import save_figure

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"


def build(world: str, data, metric_label: str) -> None:
    # Capitalized variant-type labels (leaves "5' UTR" / "3' UTR" intact).
    label_for = {s: lbl[:1].upper() + lbl[1:] for s, lbl in VEP_PANELS}

    fig, axes = plt.subplots(2, 3, figsize=figsize(FIGURE_WIDTH, 7.2), sharex=True)
    for ax, subset in zip(axes.flat, MENDELIAN_VARIANT_ORDER, strict=True):
        color = REGION_COLORS[VARIANT_REGION[subset]]
        d = data[data["subset"] == subset].sort_values("params")
        # Capless ±1 SE bars (drawn only where `se` is finite).
        ax.errorbar(
            d["params"],
            d["value"],
            yerr=d["se"],
            marker="o",
            linestyle="-",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=0,
            linewidth=1.3,
            markersize=6,
            markeredgecolor="k",
            markeredgewidth=0.4,
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_title(label_for[subset], fontsize=10)
        ax.grid(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUPRC")
    for ax in axes[-1, :]:
        ax.set_xlabel("model params", labelpad=X_LABEL_PAD)

    # Shared key: panel color encodes the variant's training-region dataset.
    regions_used = list(
        dict.fromkeys(VARIANT_REGION[s] for s in MENDELIAN_VARIANT_ORDER)
    )
    handles, labels = region_legend_handles(regions_used)
    fig.legend(
        handles,
        labels,
        title="relevant training region",
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
        title_fontsize=9,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    fig.suptitle(
        f"Parameter scaling — VEP AUPRC by variant type · {metric_label}",
        fontsize=11,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save_figure(fig, OUTPUT_DIR, f"figure5_params_vs_vep_auprc__mendelian_{world}")


def build_all() -> None:
    build("llr", ladder_llr_table(), "zero-shot LLR (new eval)")
    build("probe", ladder_probe_table(), "frozen-embedding linear probe")


if __name__ == "__main__":
    build_all()
