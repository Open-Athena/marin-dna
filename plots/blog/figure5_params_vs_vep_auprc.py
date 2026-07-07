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
import pandas as pd

from marin_dna.pipelines.evals.blog_metrics import read_llr_metrics
from plots.blog._style.figure_style import (
    EARTH_QUAL,
    FIGURE_WIDTH,
    X_LABEL_PAD,
    figsize,
)
from plots.blog._style.savefig import save_figure

DATA = Path(__file__).resolve().parent / "data" / "parameter_scaling_results.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"
LADDER_FINAL_STEP = 215573

# Subset → display label, in EARTH_QUAL color-slot order (Eric's VEP_PANELS), so a
# variant type keeps its color across Figs 5/6.
VEP_PANELS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("tss_proximal", "promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("splicing", "splicing"),
    ("synonymous_variant", "synonymous"),
)
# 1×3 task-group panels (Eric's grouping). Subset color = its VEP_PANELS slot.
FIGURE5_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CDS", ("missense_variant", "synonymous_variant")),
    ("upstream", ("tss_proximal", "5_prime_UTR_variant")),
    ("other", ("3_prime_UTR_variant", "splicing")),
)


def ladder_llr_table() -> pd.DataFrame:
    """``(params, subset, value)`` AUPRC for the 8 ladder endpoints (new Mendelian LLR)."""
    meta = pd.read_csv(DATA)[["run_name", "params"]]
    frames = []
    for _, r in meta.iterrows():
        stem = r["run_name"].removeprefix("dna-bolinas-")  # scaling-v0.5-hH-pP
        df = read_llr_metrics(
            f"{stem}-step-{LADDER_FINAL_STEP}", "mendelian_traits"
        ).to_pandas()
        df["params"] = r["params"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build() -> None:
    data = ladder_llr_table()
    color_for = {s: EARTH_QUAL[i] for i, (s, _label) in enumerate(VEP_PANELS)}
    label_for = {s: label for s, label in VEP_PANELS}

    fig, axes = plt.subplots(1, 3, figsize=figsize(FIGURE_WIDTH, 4.2))
    for ax, (group_title, subsets) in zip(axes, FIGURE5_GROUPS, strict=True):
        handles: list = []
        labels: list[str] = []
        for subset in subsets:
            d = data[data["subset"] == subset].sort_values("params")
            (line,) = ax.plot(
                d["params"],
                d["value"],
                marker="o",
                linestyle="-",
                color=color_for[subset],
                linewidth=1.3,
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.4,
                zorder=3,
            )
            handles.append(line)
            labels.append(label_for[subset])
        ax.set_xscale("log")
        ax.set_title(group_title, fontsize=10)
        ax.grid(False)
        loc = "lower right" if group_title == "CDS" else "upper left"
        ax.legend(
            handles, labels, loc=loc, fontsize=8, frameon=False, handletextpad=0.4
        )
    axes[0].set_ylabel("AUPRC")
    axes[1].set_xlabel("model params", labelpad=X_LABEL_PAD)

    fig.suptitle(
        "Parameter scaling — params vs VEP AUPRC by variant type (new eval)",
        fontsize=11,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    save_figure(fig, OUTPUT_DIR, "figure5_params_vs_vep_auprc__mendelian_llr")


if __name__ == "__main__":
    build()
