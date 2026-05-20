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
    # No legend — the timescale-legend SVG above the two panels in
    # poster.html is the canonical colour-key for both T1 and T2.
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


# ─── Comparison plots: exp21 vs Evo 2 / GPN-Star baselines ────────────
# Both plots share these baseline sources.
EVO2_METRICS_BASE = (
    "https://gist.githubusercontent.com/gonzalobenegas/"
    "3649e68fb63ca1f3443e4486078eb4d8/raw/"
    "1bce02fe0d831382d24ecbac305d401f153c65fc"
)
GPN_STAR_METRICS_BASE = (
    "https://gist.githubusercontent.com/gonzalobenegas/"
    "3649e68fb63ca1f3443e4486078eb4d8/raw/"
    "cba23a7fd89222cc72bcdddf3f37e86ee5c1075c"
)

# Per-method colour. exp21 = OA copper (the hero). Evo 2 in a cool blue/teal
# family (different from the warm OA-brand series). GPN-Star in plum.
METHOD_COLORS: dict[str, str] = {
    "exp21":       "#9e6d43",  # OA copper
    "evo2_1b":     "#7BAFC4",  # light teal
    "evo2_7b":     "#3D7A92",  # medium teal
    "evo2_40b":    "#1F4A5A",  # dark teal
    "GPN-Star-V":  "#C68DAC",  # light plum
    "GPN-Star-M":  "#8B3A62",  # OA plum
    "GPN-Star-P":  "#5A1F3F",  # dark plum
}
METHOD_LABELS: dict[str, str] = {
    "exp21":       "exp21 (promoters-yolo)",
    "evo2_1b":     "Evo 2 (1B)",
    "evo2_7b":     "Evo 2 (7B)",
    "evo2_40b":    "Evo 2 (40B)",
    "GPN-Star-V":  "GPN-Star (V)",
    "GPN-Star-M":  "GPN-Star (M)",
    "GPN-Star-P":  "GPN-Star (P)",
}

# Subsets we plot side-by-side. exp21 was trained on promoters, so the
# 5'UTR panel doubles as an off-target generalisation check.
COMPARISON_SUBSETS: dict[str, str] = {
    "Promoter":  "tss_proximal",
    "5' UTR":    "5_prime_UTR_variant",
}


def _load_exp21_trajectory() -> pl.DataFrame:
    """All exp21-promoters-yolo checkpoints (mendelian_traits)."""
    steps = (2000, 6000, 10000, 12000, 14000, 16000, 18000, 20000, 22000)
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    for step in steps:
        uri = (
            f"{S3_BASE}/exp21-promoters-yolo-step-{step}/mendelian_traits.parquet"
        )
        try:
            ck = pl.read_parquet(uri)
        except Exception as exc:
            missing.append(f"  step {step}: {exc}")
            continue
        parts.append(ck.with_columns(pl.lit(step).alias("step")))
    if missing:
        print(
            f"WARN: {len(missing)}/{len(steps)} exp21 parquets unread:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no exp21 parquets loaded"
    return pl.concat(parts)


def _baseline_values(subset: str) -> dict[str, tuple[float, float]]:
    """Per-baseline ``(AUPRC, SE)`` for one subset.

    Reads each Evo 2 model's gist parquet and the GPN-Star parquet,
    filters to the canonical score type, and returns
    ``method_id → (value, se)``.
    """
    values: dict[str, tuple[float, float]] = {}

    # Evo 2: one parquet per model, score_type "minus_llr_avg" (same
    # protocol as our marin_dna models).
    for evo_id in ("evo2_1b_base", "evo2_7b", "evo2_40b"):
        uri = f"{EVO2_METRICS_BASE}/mendelian_{evo_id}_train_metrics.parquet"
        df = pl.read_parquet(uri).filter(
            (pl.col("score_type") == "minus_llr_avg")
            & (pl.col("subset") == subset)
        )
        row = df.row(0, named=True)
        # Normalize key (drop the `_base` suffix on 1B).
        key = "evo2_1b" if evo_id == "evo2_1b_base" else evo_id
        values[key] = (float(row["value"]), float(row["se"]))

    # GPN-Star: one parquet with all 3 MSA variants, model column filters
    # them. Use the calibrated score (cLLR), the GPN-Star paper's headline.
    gpn = pl.read_parquet(f"{GPN_STAR_METRICS_BASE}/mendelian_traits.GPN-Star.parquet").filter(
        (pl.col("score_type") == "minus_llr_calibrated")
        & (pl.col("subset") == subset)
    )
    for model in ("GPN-Star-V", "GPN-Star-M", "GPN-Star-P"):
        sub = gpn.filter(pl.col("model") == model)
        if not sub.is_empty():
            row = sub.row(0, named=True)
            values[model] = (float(row["value"]), float(row["se"]))

    return values


# R1: exp21 (promoter-trained) vs Evo 2 / GPN-Star at the final checkpoint,
# on the two regions exp21 was scored against.
def plot_r1(out_path: Path) -> None:
    """Final-step bar comparison: exp21 vs Evo 2 vs GPN-Star, with SE error bars.

    Two panels (Promoter / 5'UTR). Single bar per method, ordered exp21
    first (hero), then Evo 2 by size, then GPN-Star by MSA. Error bars
    are the per-cluster bootstrap SE from the metrics parquet.
    """
    traj = _load_exp21_trajectory()
    score_type = "minus_llr_avg"
    methods = ["exp21", "evo2_1b", "evo2_7b", "evo2_40b",
               "GPN-Star-V", "GPN-Star-M", "GPN-Star-P"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, (panel, subset_key) in zip(axes, COMPARISON_SUBSETS.items()):
        # exp21 final-step (value, se)
        row = traj.filter(
            (pl.col("score_type") == score_type)
            & (pl.col("subset") == subset_key)
            & (pl.col("step") == 22000)
        ).row(0, named=True)
        exp21_final = (float(row["value"]), float(row["se"]))
        values = {"exp21": exp21_final, **_baseline_values(subset_key)}

        heights = [values[m][0] for m in methods]
        errs    = [values[m][1] for m in methods]
        colors  = [METHOD_COLORS[m] for m in methods]
        xs      = list(range(len(methods)))
        ax.bar(
            xs, heights,
            yerr=errs,
            color=colors,
            edgecolor=OA_TEXT,
            linewidth=1.0,
            error_kw={"ecolor": OA_TEXT, "elinewidth": 1.2, "capsize": 3.5},
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in methods],
            rotation=35,
            ha="right",
            fontsize=11,
        )
        if ax is axes[0]:
            ax.set_ylabel("AUPRC")
        ax.set_title(panel)
        # Value above each bar (above the error-bar cap)
        for x, h, e in zip(xs, heights, errs):
            ax.text(x, h + e + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


# ─── Entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    apply_poster_style()
    plot_t1(FIGS_DIR / "t1.svg")
    plot_t2(FIGS_DIR / "t2.svg")
    plot_r1(FIGS_DIR / "r1.svg")
