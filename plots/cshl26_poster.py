"""Plot scripts for the CSHL 2026 poster (Open Athena · AI in Biology).

Each plot function reads metrics parquets directly from S3 (so we get every
training checkpoint, not just the latest like the leaderboard parquet), applies
the Open Athena brand palette, and writes a styled SVG into the poster's
`figs/` directory where `snakemake/analysis/cshl_poster/poster.html` picks it
up via `<img src="figs/…svg">`.

Run:
    uv run python plots/cshl26_poster.py

Single file by design — see CLAUDE.md "Modularity is a means, not a goal";
this is a one-off conference artifact, not a reusable plotting library.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl


# ─── Open Athena palette ────────────────────────────────────────────────
# Mirrors openathena.ai/static/css/style.css and the Plotly defaults in the
# Delphi blog post — the same tokens the poster CSS uses for figure frames.
OA_BG               = "#BDB1A5"  # poster background (avoid in figure interior)
OA_FIG_FRAME_INNER  = "#D1C8C0"  # plot canvas bg — matches the .fig-canvas wrapper
OA_TEXT             = "#1f1e1b"
OA_TEXT_LIGHT       = "#b5aa9f"
OA_ACCENT           = "#9e6d43"  # copper — the "thing in focus" highlight

# 8-colour OA data-viz colorway (from the Delphi blog's Plotly defaults).
OA_COLORWAY = [
    "#9e6d43",  # copper
    "#2d4a3e",  # dark forest green
    "#7a3b2e",  # brick
    "#4a5d8a",  # navy blue
    "#6b5b3e",  # olive
    "#8b3a62",  # plum
    "#3d5a4f",  # teal grey
    "#a86a2c",  # burnt orange
]

# Evolutionary-timescale palette — sequential, single-hue OA copper/brown.
# The arms are inherently ordered (narrow → broad / shallow → deep in time),
# so a continuous lightness gradient encodes the ordering for free and stays
# on-brand. Same colors used in every timescale figure; exp58 just consumes
# the last 3 entries. Anchored on OA copper at mammals.
TIMESCALE_COLORS: dict[str, str] = {
    "humans":      "#D9C4A0",  # light wheat
    "primates":    "#B58853",  # warm tan
    "mammals":     "#9e6d43",  # OA copper (accent)
    "vertebrates": "#6E4421",  # dark coffee
    "animals":     "#3D2417",  # very dark espresso
}
ARM_LABEL: dict[str, str] = {
    "humans":      "humans (1 sp.)",
    "primates":    "primates (~65 Mya, 11 sp.)",
    "mammals":     "mammals (~100 Mya, 81 sp.)",
    "vertebrates": "vertebrates (~600 Mya, 317 sp.)",
    "animals":     "animals (~800 Mya, 499 sp.)",
}


def apply_poster_style() -> None:
    """matplotlib rcParams for figures that live inside the poster frames."""
    mpl.rcParams.update({
        # Render text as SVG <path>s so the figure is self-contained — no
        # dependency on Lato being installed wherever the poster is viewed.
        "svg.fonttype":        "path",
        "font.family":         "sans-serif",
        "font.size":           14,
        "axes.titlesize":      16,
        "axes.labelsize":      14,
        "axes.titleweight":    "normal",
        "axes.edgecolor":      OA_TEXT,
        "axes.labelcolor":     OA_TEXT,
        "axes.linewidth":      1.5,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.facecolor":      OA_FIG_FRAME_INNER,
        "figure.facecolor":    OA_FIG_FRAME_INNER,
        "xtick.color":         OA_TEXT,
        "ytick.color":         OA_TEXT,
        "xtick.labelsize":     12,
        "ytick.labelsize":     12,
        "legend.frameon":      False,
        "legend.fontsize":     12,
        "lines.linewidth":     2.5,
        "lines.markersize":    7,
        "savefig.facecolor":   OA_FIG_FRAME_INNER,
        "savefig.bbox":        "tight",
    })


# ─── Data ──────────────────────────────────────────────────────────────
S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"

# poster.html references figures as figs/<stem>.svg
FIGS_DIR = (
    Path(__file__).parent.parent
    / "snakemake" / "analysis" / "cshl_poster" / "figs"
)


def load_exp55(arms: tuple[str, ...], steps: tuple[int, ...]) -> pl.DataFrame:
    """Read mendelian_traits metrics across exp55-{arm}-step-{step}."""
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    for arm in arms:
        for step in steps:
            uri = f"{S3_BASE}/exp55-{arm}-step-{step}/mendelian_traits.parquet"
            try:
                df = pl.read_parquet(uri)
            except Exception as exc:
                missing.append(f"  {arm} step {step}: {exc}")
                continue
            parts.append(
                df.with_columns(
                    pl.lit(arm).alias("arm"),
                    pl.lit(step).alias("step"),
                )
            )
    if missing:
        print(
            f"WARN: {len(missing)}/{len(arms) * len(steps)} parquets unread:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no parquets loaded — has the sweep started?"
    return pl.concat(parts)


def _plot_timescale_panel(
    df: pl.DataFrame,
    arms: tuple[str, ...],
    *,
    out_path: Path,
) -> None:
    """Shared plot body for the timescale figures (T1 / T2).

    No internal title — the surrounding poster panel already has a
    `<p class="fig-title">` header; an in-SVG title would duplicate it.
    """
    # Aspect ratio 1.6:1 matches the placeholder figs and the poster's
    # .fig-canvas wrapper, avoiding letterboxing on print.
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in arms:
        arm_df = df.filter(pl.col("arm") == arm).sort("step")
        if arm_df.is_empty():
            continue
        ax.plot(
            arm_df["step"].to_numpy(),
            arm_df["value"].to_numpy(),
            marker="o",
            color=TIMESCALE_COLORS[arm],
            label=ARM_LABEL[arm],
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("AUPRC")
    # Legend ordering follows `arms`, which is already narrow → broad —
    # so the legend doubles as a colour-key for the sequential palette.
    ax.legend(loc="lower right", title=None, labelcolor=OA_TEXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


# ─── T1: promoter AUPRC vs evolutionary timescale (exp55) ──────────────
def plot_t1(out_path: Path) -> None:
    """Promoter AUPRC across the exp55 timescale arms (mammals peaks)."""
    arms = ("humans", "primates", "mammals", "vertebrates", "animals")
    steps = (1000, 2000, 3000, 4000, 5000, 9000, 13000, 16999)
    score_type = "minus_llr_avg"   # canonical scoring for mendelian promoters
    subset = "tss_proximal"        # promoter consequence subset

    raw = load_exp55(arms, steps)
    df = raw.filter(
        (pl.col("score_type") == score_type)
        & (pl.col("subset") == subset)
    )
    assert not df.is_empty(), (
        f"empty after filter on {score_type}/{subset}; "
        f"score_types={sorted(raw['score_type'].unique().to_list())}, "
        f"subsets={sorted(raw['subset'].unique().to_list())}"
    )
    _plot_timescale_panel(df, arms, out_path=out_path)


# ─── T2: CDS (missense) AUPRC vs evolutionary timescale (exp58) ────────  # noqa: E501
def plot_t2(out_path: Path) -> None:
    """Missense AUPRC across the exp58 timescale arms (animals optimum)."""
    arms = ("mammals", "vertebrates", "animals")
    steps = (1000, 2000, 3000, 4000, 5000, 9000, 13000, 16999)
    score_type = "minus_llr_avg"
    subset = "missense_variant"

    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    for arm in arms:
        for step in steps:
            uri = f"{S3_BASE}/exp58-{arm}-step-{step}/mendelian_traits.parquet"
            try:
                ck = pl.read_parquet(uri)
            except Exception as exc:
                missing.append(f"  {arm} step {step}: {exc}")
                continue
            parts.append(
                ck.with_columns(
                    pl.lit(arm).alias("arm"),
                    pl.lit(step).alias("step"),
                )
            )
    if missing:
        print(
            f"WARN: {len(missing)}/{len(arms) * len(steps)} parquets unread:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no exp58 parquets loaded"
    raw = pl.concat(parts)
    df = raw.filter(
        (pl.col("score_type") == score_type)
        & (pl.col("subset") == subset)
    )
    assert not df.is_empty(), (
        f"empty after filter on {score_type}/{subset}; "
        f"score_types={sorted(raw['score_type'].unique().to_list())}, "
        f"subsets={sorted(raw['subset'].unique().to_list())}"
    )
    _plot_timescale_panel(df, arms, out_path=out_path)


# ─── Entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    apply_poster_style()
    plot_t1(FIGS_DIR / "t1.svg")
    plot_t2(FIGS_DIR / "t2.svg")
