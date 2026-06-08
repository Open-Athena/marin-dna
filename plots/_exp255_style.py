"""Shared visual style + palette for the exp255 (#255) figures (import-only helper).

Not a recipe — the `_` prefix marks it private. Each exp255 plot calls `set_style()`
once and pulls the cohort colors from here so the whole #255 figure set looks uniform.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COL_FAMILY = "#8c98a4"  # 108-sp family — muted slate (the baseline)
COL_ORDER = "#1f6fb2"  # 19-sp order — confident blue (the new arm)
COL_ACCENT = "#d1495b"  # highlight for the ncRNA / 5'UTR family edge
GRID = "#e2e2e2"
BASELINE = 0.10  # Mendelian prevalence baseline (1:9 matching)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5a5a5a",
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": "#444",
            "ytick.color": "#444",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str, name: str | None = None) -> Path:
    """Write `output/<stem>/<name or stem>.{png,svg}`, close the figure, return the PNG path."""
    out = Path(__file__).parent / "output" / stem
    out.mkdir(parents=True, exist_ok=True)
    name = name or stem
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    return out / f"{name}.png"
