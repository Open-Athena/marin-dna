"""Figure 11 (M·LLR world): Mendelian VEP leaderboard heatmap, AUPRC %.

Redo of the blog's Figure 11, in Eric's style but driven from **live** evals_v2
metrics (the new Mendelian eval) via
``marin_dna.pipelines.evals.leaderboard.normalized_rows`` — instead of the
hand-extracted CSV snapshot in Eric's post repo. Each family is shown at its
``DEFAULT_PROTOCOL`` (the canonical zero-shot number the leaderboard renders):
marin_dna → LLR, gpn_star → cLLR, conservation → score, alphagenome → L2,
evo2 → LLR. Rows are models ordered by Macro Avg; columns are the headline Macro
Avg plus each per-subset AUPRC. The Macro Avg column is boxed, and our models'
(``marin_dna``) row labels are bolded.

This is the #362 end-to-end proof that the ported style + the live-data path
render a blog-faithful figure. Headline model curation across the family and the
other three worlds (M·Probe / S·LLR / S·Probe) is #363.

Run:  uv run python -m plots.blog.figure11_leaderboard_heatmap
Out:  plots/output/blog/figure11_leaderboard_heatmap__mendelian_llr.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from marin_dna.pipelines.evals.leaderboard import DEFAULT_PROTOCOL, normalized_rows
from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET
from plots.blog._style.figure_style import HEATMAP_CMAP, figsize
from plots.blog._style.savefig import save_figure

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

# Raw consequence subset → display label, in leaderboard column order.
# Mirrors SUBSET_DISPLAY in dashboard/src/components/heatmap.js.
SUBSET_DISPLAY: dict[str, str] = {
    "missense_variant": "Missense",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5' UTR",
    "tss_proximal": "Promoter",
    "non_coding_transcript_exon_variant": "ncRNA",
    "3_prime_UTR_variant": "3' UTR",
    "distal": "Distal",
    "synonymous_variant": "Synonymous",
}
SUBSET_ORDER = list(SUBSET_DISPLAY)


def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build() -> None:
    rows = normalized_rows("mendelian_traits")

    # Each family at its default (canonical zero-shot) protocol — the leaderboard view.
    rows = rows.with_columns(
        pl.col("family")
        .replace_strict(DEFAULT_PROTOCOL, default=None)
        .alias("_default")
    ).filter(pl.col("protocol") == pl.col("_default"))

    # Pivot (method × subset) → AUPRC %, keeping the macro aggregate + the 8 subsets.
    keep = [MACRO_AVG_SUBSET, *SUBSET_ORDER]
    pdf = (
        rows.filter(pl.col("subset").is_in(keep))
        .with_columns(value=pl.col("value") * 100.0)
        .select(["method_display", "family", "subset", "value"])
        .to_pandas()
    )
    table = pdf.pivot_table(
        index=["method_display", "family"], columns="subset", values="value"
    ).reindex(columns=keep)
    table = table.sort_values(MACRO_AVG_SUBSET, ascending=False)

    disp = [idx[0] for idx in table.index]
    fams = [idx[1] for idx in table.index]
    col_labels = ["Macro Avg", *[SUBSET_DISPLAY[s] for s in SUBSET_ORDER]]
    matrix = table.to_numpy(dtype=float)
    n, m = matrix.shape

    finite = matrix[np.isfinite(matrix)]
    norm = Normalize(finite.min(), finite.max())
    cmap = HEATMAP_CMAP

    fig, ax = plt.subplots(figsize=figsize(10.0, max(4.4, 0.30 * n)))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # White gridlines for a clean tiled look.
    ax.set_xticks(np.arange(-0.5, m), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    # Cell value annotations (Macro Avg column in bold; blank cells left empty).
    for i in range(n):
        for j in range(m):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            tc = "white" if _luminance(cmap(norm(v))) < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=tc,
                fontweight="bold" if j == 0 else "normal",
            )

    # Box the Macro Avg column.
    ax.add_patch(
        Rectangle((-0.5, -0.5), 1, n, fill=False, edgecolor="black", lw=2.0, zorder=5)
    )

    ax.set_xticks(range(m))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(disp, fontsize=9)
    for tick, fam in zip(ax.get_yticklabels(), fams):
        if fam == "marin_dna":
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.02
    )
    cb.set_label("AUPRC (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(
        "Mendelian VEP benchmark — AUPRC (%) · zero-shot LLR (new eval)",
        fontsize=9.5,
        pad=10,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, "figure11_leaderboard_heatmap__mendelian_llr")


if __name__ == "__main__":
    build()
