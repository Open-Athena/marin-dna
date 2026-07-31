"""Shared table builder and renderer for Mendelian leaderboard heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from marin_dna.blog_figure_typography import (
    MATPLOTLIB_NOTE_SIZE,
    normalize_matplotlib_svg_typography_file,
)
from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET
from plots.blog.genomic_lm_optimization.src.utils import figure_theme as _figure_theme  # noqa: F401
from plots.blog.genomic_lm_optimization.src.utils.figure_style import (
    HEATMAP_CMAP,
    figsize,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

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


def _luminance(rgba: tuple[float, ...] | np.ndarray) -> float:
    r, g, b = rgba[:3]
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def table_from_normalized(rows: pl.DataFrame) -> pd.DataFrame:
    """Convert normalized leaderboard rows to an AUPRC-percent heatmap table."""
    keep = [MACRO_AVG_SUBSET, *SUBSET_ORDER]
    selected = (
        rows.filter(pl.col("subset").is_in(keep))
        .with_columns(value=pl.col("value") * 100.0)
        .select(["method_display", "family", "subset", "value"])
    )
    duplicates = (
        selected.group_by(["method_display", "family", "subset"])
        .len()
        .filter(pl.col("len") != 1)
    )
    assert duplicates.height == 0, f"duplicate leaderboard cells:\n{duplicates}"
    pdf = selected.to_pandas()
    return pdf.pivot(
        index=["method_display", "family"], columns="subset", values="value"
    ).reindex(columns=keep)


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Write deterministic SVG plus local-review PNG and PDF artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with mpl.rc_context({"svg.hashsalt": name}):
        for extension, options in (
            ("png", {"dpi": 300}),
            ("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
            ("svg", {"metadata": {"Date": None}}),
        ):
            path = OUTPUT_DIR / f"{name}.{extension}"
            fig.savefig(path, bbox_inches="tight", transparent=True, **options)
            if extension == "svg":
                lines = path.read_text().splitlines()
                path.write_text("\n".join(line.rstrip() for line in lines) + "\n")
                normalize_matplotlib_svg_typography_file(path)
            paths.append(path)
    print("Wrote " + ", ".join(str(path) for path in paths))


def render_heatmap(
    table: pd.DataFrame,
    *,
    output_name: str,
) -> None:
    """Render one independently sorted and color-normalized leaderboard."""
    table = table.sort_values(MACRO_AVG_SUBSET, ascending=False)
    display_names = [index[0] for index in table.index]
    families = [index[1] for index in table.index]
    column_labels = ["Macro Avg", *SUBSET_DISPLAY.values()]
    matrix = table[[MACRO_AVG_SUBSET, *SUBSET_ORDER]].to_numpy(dtype=float)
    n_rows, n_columns = matrix.shape

    finite = matrix[np.isfinite(matrix)]
    assert finite.size > 0, "leaderboard has no finite values"
    norm = Normalize(float(finite.min()), float(finite.max()))

    fig, ax = plt.subplots(figsize=figsize(10.0, max(4.4, 0.30 * n_rows)))
    ax.imshow(matrix, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(-0.5, n_columns), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    for row_index in range(n_rows):
        for column_index in range(n_columns):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                continue
            text_color = (
                "white" if _luminance(HEATMAP_CMAP(norm(value))) < 0.5 else "black"
            )
            ax.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=MATPLOTLIB_NOTE_SIZE,
                color=text_color,
                fontweight="bold" if column_index == 0 else "normal",
            )

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            1,
            n_rows,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            zorder=5,
        )
    )
    ax.set_xticks(range(n_columns))
    ax.set_xticklabels(column_labels, rotation=30, ha="right")
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(display_names)
    for tick, family in zip(ax.get_yticklabels(), families, strict=True):
        if family == "marin_dna":
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=HEATMAP_CMAP),
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("AUPRC (%)")

    fig.tight_layout()
    _save_figure(fig, output_name)
    plt.close(fig)
