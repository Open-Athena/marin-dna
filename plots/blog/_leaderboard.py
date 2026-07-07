"""Shared builder for the Mendelian leaderboard heatmaps (Fig 11 × worlds).

Renders a ``models × (Macro Avg + per-consequence-subset)`` AUPRC-% heatmap in
Eric's blog style (`_style/`). One ``render_heatmap`` is reused by every world;
``table_from_normalized`` turns the tidy long-form leaderboard rows
(``normalized_rows`` / ``probe_normalized_rows`` — same schema) into the pandas
table it expects. SGE (a per-accession grid) has its own prep — see the SGE recipe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

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


def table_from_normalized(rows: pl.DataFrame) -> pd.DataFrame:
    """Long-form ``normalized_rows`` / ``probe_normalized_rows`` → the pandas
    heatmap table (index ``(method_display, family)``, columns
    ``[_macro_avg_, *SUBSET_ORDER]``, values AUPRC %)."""
    keep = [MACRO_AVG_SUBSET, *SUBSET_ORDER]
    pdf = (
        rows.filter(pl.col("subset").is_in(keep))
        .with_columns(value=pl.col("value") * 100.0)
        .select(["method_display", "family", "subset", "value"])
        .to_pandas()
    )
    return pdf.pivot_table(
        index=["method_display", "family"], columns="subset", values="value"
    ).reindex(columns=keep)


def render_heatmap(table: pd.DataFrame, *, title: str, out_name: str) -> None:
    """Render one leaderboard heatmap (Eric's style) and save SVG/PNG/PDF.

    ``table``: index ``(method_display, family)``, columns ``[_macro_avg_,
    *SUBSET_ORDER]``, values AUPRC %. Rows are sorted by Macro Avg; the Macro Avg
    column is boxed; ``marin_dna`` row labels are bolded; blank cells are left empty.
    """
    table = table.sort_values(MACRO_AVG_SUBSET, ascending=False)
    disp = [idx[0] for idx in table.index]
    fams = [idx[1] for idx in table.index]
    col_labels = ["Macro Avg", *[SUBSET_DISPLAY[s] for s in SUBSET_ORDER]]
    matrix = table[[MACRO_AVG_SUBSET, *SUBSET_ORDER]].to_numpy(dtype=float)
    n, m = matrix.shape

    finite = matrix[np.isfinite(matrix)]
    norm = Normalize(finite.min(), finite.max())
    cmap = HEATMAP_CMAP

    fig, ax = plt.subplots(figsize=figsize(10.0, max(4.4, 0.30 * n)))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(-0.5, m), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

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

    ax.set_title(title, fontsize=9.5, pad=10)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, out_name)
