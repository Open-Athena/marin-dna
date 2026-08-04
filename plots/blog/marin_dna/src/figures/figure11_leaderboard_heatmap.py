"""Figure 11: Mendelian VEP benchmark heatmap (AUPRC %).

Rows are models ordered by Macro Avg; columns are the headline Macro Avg (over 8
subsets) plus each per-subset AUPRC. The Macro Avg column is boxed and bolded as
the summary metric, and our model's row label is bolded. Data come from
data/model_leaderboard.csv (openathena.ai Mendelian leaderboard); the 'Global'
column is dropped.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from figures.data import save
from utils.figure_style import HEATMAP_CMAP, figsize

MACRO = "Macro Avg (8 subsets)"
SUBSETS = [
    "Missense",
    "Splicing",
    "5' UTR",
    "Promoter",
    "ncRNA",
    "3' UTR",
    "Distal",
    "Synonymous",
]

# (CSV model name, display label). Rows are re-sorted by Macro Avg below.
MODELS: tuple[tuple[str, str], ...] = (
    ("GPN-Star (M)", "GPN-Star (M)"),
    ("AlphaGenome", "AlphaGenome"),
    ("exp135-1B-m5.1", "MarinDNA (1B/m5.1)"),
    ("Evo 2 (40B)", "Evo 2 (40B)"),
    ("Evo 2 (7B)", "Evo 2 (7B)"),
    ("Evo 2 (1B base)", "Evo 2 (1B base)"),
)
HIGHLIGHT_ROW = "MarinDNA (1B/m5.1)"


def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build(df: pd.DataFrame) -> None:
    df = df.set_index("Model")
    labels = {csv: disp for csv, disp in MODELS}
    names = sorted((csv for csv, _ in MODELS), key=lambda c: -float(df.loc[c, MACRO]))
    disp = [labels[c] for c in names]

    cols = [MACRO] + SUBSETS
    col_labels = ["Macro Avg"] + SUBSETS
    M = df.loc[names, cols].to_numpy(dtype=float)
    n, m = M.shape

    norm = Normalize(M.min(), M.max())
    cmap = HEATMAP_CMAP

    fig, ax = plt.subplots(figsize=figsize(10.0, 4.4))
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

    # White gridlines for a clean tiled look.
    ax.set_xticks(np.arange(-0.5, m), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    # Cell value annotations (Macro Avg column in bold).
    for i in range(n):
        for j in range(m):
            v = M[i, j]
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

    # Highlight the Macro Avg column.
    ax.add_patch(
        Rectangle((-0.5, -0.5), 1, n, fill=False, edgecolor="black", lw=2.0, zorder=5)
    )

    ax.set_xticks(range(m))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(disp, fontsize=9)
    for t in ax.get_yticklabels():
        if t.get_text() == HIGHLIGHT_ROW:
            t.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.02
    )
    cb.set_label("AUPRC (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Authored narrower than the other figures (so it scales up more at the
    # column width); a smaller title keeps it visually in line with the rest.
    ax.set_title("Mendelian VEP benchmark — AUPRC (%)", fontsize=9.5, pad=10)
    fig.tight_layout()
    save(fig, "figure11_leaderboard_heatmap")
