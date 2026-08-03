"""Headline zero-shot VEP performance versus as-deployed inference cost."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from figures.data import save
from utils.figure_style import (
    FIGURE_WIDTH,
    MODEL_FAMILY_MARKERS,
    SCORING_PROTOCOL_COLORS,
    figsize,
    set_square_subplot_height,
)

SUBPLOT_HEIGHT_PX = 160.0
MACRO_COLUMN = "Macro Avg (8 subsets)"
MODEL_NAMES = (
    "exp135-1B-m5.1",
    "Evo 2 (1B base)",
    "Evo 2 (7B)",
    "Evo 2 (40B)",
)
DISPLAY_NAMES = {
    "exp135-1B-m5.1": "MarinDNA 1B",
    "Evo 2 (1B base)": "Evo 2 1B",
    "Evo 2 (7B)": "Evo 2 7B",
    "Evo 2 (40B)": "Evo 2 40B",
}
MARINDNA_COLOR = SCORING_PROTOCOL_COLORS["llr"]
EVO2_COLOR = "#8a7d69"


def _cost_tick(value: float, _position: float) -> str:
    """Format automatically located log-scale ticks as dollar amounts."""
    if value >= 1_000:
        return f"${value / 1_000:g}k"
    if value >= 1:
        return f"${value:g}"
    return f"${value:.1g}"


def _headline_data(
    leaderboard: pd.DataFrame,
    inference_costs: pd.DataFrame,
) -> pd.DataFrame:
    assert leaderboard["Model"].is_unique
    assert inference_costs["Model"].is_unique
    assert set(inference_costs["Model"]) == set(MODEL_NAMES)
    selected = leaderboard[leaderboard["Model"].isin(MODEL_NAMES)][
        ["Model", MACRO_COLUMN]
    ]
    assert set(selected["Model"]) == set(MODEL_NAMES)
    data = inference_costs.merge(selected, on="Model", validate="one_to_one")
    assert len(data) == len(MODEL_NAMES)
    assert data["Cost (USD / 1M variants)"].gt(0).all()
    assert data[MACRO_COLUMN].between(0, 100).all()
    return data


def build(leaderboard: pd.DataFrame, inference_costs: pd.DataFrame) -> None:
    data = _headline_data(leaderboard, inference_costs)
    evo2 = data[data["Family"] == "Evo 2"].sort_values("Parameters (B)")
    marindna = data[data["Family"] == "MarinDNA"]
    assert len(evo2) == 3
    assert len(marindna) == 1

    fig, ax = plt.subplots(figsize=figsize(FIGURE_WIDTH * 0.72, 7.2))
    ax.plot(
        evo2["Cost (USD / 1M variants)"],
        evo2[MACRO_COLUMN],
        color=EVO2_COLOR,
        marker=MODEL_FAMILY_MARKERS["evo2"],
        markeredgecolor="#1f1e1b",
        label="Evo 2",
        zorder=2,
    )
    ax.plot(
        marindna["Cost (USD / 1M variants)"],
        marindna[MACRO_COLUMN],
        color=MARINDNA_COLOR,
        marker=MODEL_FAMILY_MARKERS["marindna"],
        markeredgecolor="#1f1e1b",
        linestyle="none",
        label="MarinDNA",
        zorder=3,
    )

    label_offsets = {
        "exp135-1B-m5.1": (8, 7, "left", "bottom"),
        "Evo 2 (1B base)": (7, -7, "left", "top"),
        "Evo 2 (7B)": (7, 6, "left", "bottom"),
        "Evo 2 (40B)": (-8, -7, "right", "top"),
    }
    for _, row in data.iterrows():
        model = str(row["Model"])
        score = float(row[MACRO_COLUMN])
        dx, dy, ha, va = label_offsets[model]
        ax.annotate(
            DISPLAY_NAMES[model],
            xy=(float(row["Cost (USD / 1M variants)"]), score),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=MARINDNA_COLOR if model == "exp135-1B-m5.1" else EVO2_COLOR,
            fontweight="bold" if model == "exp135-1B-m5.1" else "normal",
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_cost_tick))
    ax.set_xlabel("Cost per 1M variants (USD)")
    ax.set_ylabel("VEP AUPRC (%)")
    ax.grid(which="major", color="#d9ccba", linewidth=0.8)
    ax.margins(x=0.12, y=0.28)
    ax.set_box_aspect(1)

    fig.tight_layout()
    set_square_subplot_height(fig, (ax,), SUBPLOT_HEIGHT_PX)
    save(fig, "headline_cost_performance")
