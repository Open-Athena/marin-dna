"""Figure 6 (M·LLR): parameter scaling — validation loss vs VEP AUPRC by variant type.

Redo of the blog's Fig 6 (Eric's 2×3 scatter) on the **new** Mendelian eval:
x = final validation loss (vendored scaling CSV), y = new-eval AUPRC
(``blog_metrics``), marker color = params. Dotted linear best-fit + Pearson ρ per
panel make the blog's point — lower loss does **not** cleanly predict better VEP.

Run:  uv run python -m plots.blog.figure6_loss_vs_vep_auprc
Out:  plots/output/blog/figure6_loss_vs_vep_auprc__mendelian_llr.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def build(world: str, data, metric_label: str) -> None:
    params_present = sorted({int(p) for p in data["params"].unique()})
    pal = palette(params_present)

    fig, axes = plt.subplots(2, 3, figsize=figsize(FIGURE_WIDTH, 7.0))
    for ax, (subset, title) in zip(axes.flat, VEP_PANELS, strict=True):
        d = data[data["subset"] == subset].sort_values("params")
        n = int(d["n"].iloc[0]) if len(d) else 0
        for _, row in d.iterrows():
            ax.scatter(
                row["eval_loss"],
                row["value"],
                s=_MARKER_AREA,
                color=pal[int(row["params"])],
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
            )
        if len(d) >= 2:
            xs = d["eval_loss"].to_numpy(dtype=float)
            ys = d["value"].to_numpy(dtype=float)
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
            rho = float(np.corrcoef(xs, ys)[0, 1])
            ax.text(
                0.97,
                0.95,
                rf"$\rho={rho:.2f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.35",
            )
        ax.set_title(f"{title} (n={n:,})", fontsize=10)
        ax.grid(False)
        ax.margins(y=0.12)

    axes[-1, 1].set_xlabel("loss", labelpad=X_LABEL_PAD)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUPRC")

    fig.suptitle(
        f"Parameter scaling — loss vs VEP AUPRC by variant type · {metric_label}",
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
