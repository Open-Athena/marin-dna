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
import numpy as np
import polars as pl
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge


# ─── Open Athena palette ────────────────────────────────────────────────
# Mirrors openathena.ai/static/css/style.css and the Plotly defaults in the
# Delphi blog post — the same tokens the poster CSS uses for figure frames.
OA_BG               = "#BDB1A5"  # poster background (avoid in figure interior)
OA_FIG_FRAME_INNER  = "#D1C8C0"  # plot canvas bg — matches the .fig-canvas wrapper
OA_TEXT             = "#1f1e1b"
OA_TEXT_LIGHT       = "#b5aa9f"
OA_ACCENT           = "#9e6d43"  # copper — the "thing in focus" highlight

# ─── Font sizes (shared between bar plot and line plots) ────────────────
# Single source of truth so the R1 specialist bar chart and the R2 / R3
# / T1 / T2 line plots all use the same per-element font scale. Per-fig
# overrides applied with these constants (rather than the apply_poster_style
# rcParams) because the bar chart was originally tuned at smaller sizes
# and bumping the rcParams baseline would also affect other plots.
POSTER_TITLE_FS  = 32
POSTER_LABEL_FS  = 30
POSTER_TICK_FS   = 26
POSTER_LEGEND_FS = 32

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

# Evolutionary-timescale palette — `viridis` (perceptually uniform,
# colorblind-safe sequential). Lightest = most recent (humans), darkest
# = oldest (animals). Standard matplotlib palette so the ordering reads
# clearly to anyone who's seen a scientific figure before. Chosen so the
# *mixture* palette can be a different sequential (magma) — same family
# of cmaps but visually distinct.
TIMESCALE_COLORS: dict[str, str] = {
    "humans":      "#90d743",  # viridis[4]  — yellow-green
    "primates":    "#35b779",  # viridis[3]  — green
    "mammals":     "#21918c",  # viridis[2]  — teal
    "vertebrates": "#31688e",  # viridis[1]  — blue
    "animals":     "#443983",  # viridis[0]  — dark purple
}
ARM_LABEL: dict[str, str] = {
    "humans":      "humans (1 sp.)",
    "primates":    "primates (~65 Mya, 11 sp.)",
    "mammals":     "mammals (~100 Mya, 81 sp.)",
    "vertebrates": "vertebrates (~600 Mya, 317 sp.)",
    "animals":     "animals (~800 Mya, 499 sp.)",
}


def apply_poster_style() -> None:
    """matplotlib rcParams for figures that live inside the poster frames.

    Font sizes set in *matplotlib points* at the figsize used by each
    plot function. The SVG is then scaled when embedded in the column.
    Sizes here picked so the rendered text on the printed 44 × 44 in
    poster reads at ~24-28pt physical — matching the HTML body text
    (--fs-body / --fs-caption in poster.html).

    Plot backgrounds are WHITE (standard data-viz convention), not the
    OA cream. The cream chrome lives one level up in the `.fig-canvas`
    wrapper around each plot. Chrome stays OA brand; data area is
    standard high-contrast.
    """
    mpl.rcParams.update({
        # Render text as SVG <path>s so the figure is self-contained — no
        # dependency on Lato being installed wherever the poster is viewed.
        "svg.fonttype":        "path",
        "font.family":         "sans-serif",
        "font.size":           18,
        "axes.titlesize":      22,
        "axes.labelsize":      20,
        "axes.titleweight":    "normal",
        "axes.edgecolor":      OA_TEXT,
        "axes.labelcolor":     OA_TEXT,
        "axes.linewidth":      1.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.facecolor":      "white",
        "figure.facecolor":    "white",
        "xtick.color":         OA_TEXT,
        "ytick.color":         OA_TEXT,
        "xtick.labelsize":     16,
        "ytick.labelsize":     16,
        "legend.frameon":      False,
        "legend.fontsize":     16,
        # Thick lines + chunky markers so the line plots read at ~1m
        # viewing distance on the printed 44x44in poster. Applies to all
        # ax.plot() calls (T1, T2, R3) — keeping these consistent.
        "lines.linewidth":     4.5,
        "lines.markersize":    14,
        "savefig.facecolor":   "white",
        "savefig.bbox":        "tight",
    })


# ─── Data ──────────────────────────────────────────────────────────────
S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"

# Poster source (poster.typ) lives at docs/posters/cshl26/ and references
# the figures as figs/<stem>.svg. We write the matplotlib outputs there
# directly so a `typst compile poster.typ` finds them.
FIGS_DIR = (
    Path(__file__).parent.parent
    / "docs" / "posters" / "cshl26" / "figs"
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
    apply_poster_style()
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


def _load_exp_timescale(
    exp_id: str, arms: tuple[str, ...], steps: tuple[int, ...],
) -> pl.DataFrame:
    """Read mendelian_traits metrics across {exp_id}-{arm}-step-{step}.

    Shared loader for the exp55 / exp58 / exp59 timescale sweeps. Each
    parquet on S3 has the same schema; only the experiment id and the
    arm names vary.
    """
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    for arm in arms:
        for step in steps:
            uri = f"{S3_BASE}/{exp_id}-{arm}-step-{step}/mendelian_traits.parquet"
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
            f"WARN: {len(missing)}/{len(arms) * len(steps)} {exp_id} parquets unread:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, f"no {exp_id} parquets loaded"
    return pl.concat(parts)


def plot_timescale(out_path: Path) -> None:
    """T1 + T2 + T3 in a single 3-panel figure (Promoter | Missense | 3' UTR).

    Mirrors the structure of plot_r3 (multi-panel line plot with
    colour-coded subplot titles). Saves vertical poster space vs
    stacking three separate SVGs, and makes the "same dataset family
    seen through three consequence lenses" framing visible at a glance.
    """
    apply_poster_style()

    promoter_arms = ("humans", "primates", "mammals", "vertebrates", "animals")
    cds_arms      = ("mammals", "vertebrates", "animals")
    utr3_arms     = ("mammals", "vertebrates", "animals")
    # Prototype: uniformly-spaced steps (~4000 apart) instead of the
    # original dense-early sampling (1000…5000 every 1k, then 9k/13k/17k).
    # The dense early sampling was useful for seeing fast dynamics, but
    # for a poster figure the uniform spacing reads more cleanly.
    # To restore dense-early: (1000, 2000, 3000, 4000, 5000, 9000, 13000, 16999)
    steps         = (1000, 5000, 9000, 13000, 16999)
    score_type    = "minus_llr_avg"

    # Promoter (exp55) — TSS-proximal subset.
    promoter_df = _load_exp_timescale("exp55", promoter_arms, steps).filter(
        (pl.col("score_type") == score_type)
        & (pl.col("subset") == "tss_proximal")
    )
    assert not promoter_df.is_empty(), "empty promoter timescale df"

    # CDS (exp58) — missense subset.
    cds_df = _load_exp_timescale("exp58", cds_arms, steps).filter(
        (pl.col("score_type") == score_type)
        & (pl.col("subset") == "missense_variant")
    )
    assert not cds_df.is_empty(), "empty CDS timescale df"

    # 3' UTR (exp59) — 3_prime_UTR_variant subset.
    utr3_df = _load_exp_timescale("exp59", utr3_arms, steps).filter(
        (pl.col("score_type") == score_type)
        & (pl.col("subset") == "3_prime_UTR_variant")
    )
    assert not utr3_df.is_empty(), "empty 3' UTR timescale df"

    title_fs  = POSTER_TITLE_FS
    label_fs  = POSTER_LABEL_FS
    tick_fs   = POSTER_TICK_FS

    # 3 panels (was 2) — figure stays at total width 15 but height
    # bumped a touch so each panel doesn't get aspect-squished too far.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    # 3' UTR title colour: a darker tan that nods to the beige UTR
    # block in the gene-cartoon schematic (figs/region_legend.svg)
    # without being so light it's unreadable on white.
    utr3_title_color = "#8a6d3b"

    panel_data = (
        ("Promoter variants",   promoter_arms, promoter_df,
            REGION_TITLE_COLORS["Promoter variants"]),
        ("Missense variants",   cds_arms,      cds_df,
            REGION_TITLE_COLORS["Missense variants"]),
        ("3' UTR variants",     utr3_arms,     utr3_df,
            utr3_title_color),
    )
    for ax, (panel, arms, df, title_color) in zip(axes, panel_data):
        for arm in arms:
            sub = df.filter(pl.col("arm") == arm).sort("step")
            if sub.is_empty():
                continue
            ax.plot(
                sub["step"].to_numpy(),
                sub["value"].to_numpy(),
                marker="o",
                color=TIMESCALE_COLORS[arm],
                label=ARM_LABEL[arm],
            )
        ax.set_xlabel("training step", fontsize=label_fs)
        if ax is axes[0]:
            ax.set_ylabel("AUPRC", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        # 3 panels in figsize=(15, 4.5) → ~5in/panel; at tick_fs=26
        # the auto-locator's 5000-step ticks ("5000 10000 15000") run
        # into each other. Cap to ~3 ticks so they get nicer spacing.
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.set_title(
            panel,
            fontweight="bold",
            fontsize=title_fs,
            color=title_color,
        )

    # No in-plot legend — the timescale_legend.svg schematic above the
    # plot is the canonical colour-key for all panels (clade colours
    # mirror the bars there).
    fig.subplots_adjust(bottom=0.15, top=0.92, left=0.06, right=0.98, wspace=0.25)

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

# Per-method colour for the exp13 mixture sweep.
#
# The 4 mixture conditions sit on a 1-D axis (fraction of CDS in the
# training data: 0 % → 50 % → 90 % → 100 %). We anchor the endpoints
# at the focal region colours — Promoter blue at 0 % CDS, CDS orange
# at 100 % — and linearly interpolate in RGB for the two intermediate
# mixtures. The colour itself thus encodes the recipe.
METHOD_COLORS: dict[str, str] = {
    "exp21":                  "#0173b2",  # 100 % promoter — Promoter blue
    "exp13-equal":            "#6f815b",  # 50 % CDS       — midpoint (olive)
    "exp13-proportional":     "#c78c16",  # 90 % CDS       — 9/10 toward orange
    "exp27":                  "#de8f05",  # 100 % CDS      — CDS orange
}
METHOD_LABELS: dict[str, str] = {
    "exp21":                  "Promoter only",
    "exp27":                  "CDS only",
    # Parenthetical ratios duplicate the pie chart drawn next to each
    # legend entry — kept both for redundancy: the pie reads at a glance
    # for visual comparison, the numbers nail the exact ratio.
    "exp13-equal":            "Uniform mix (50 / 50)",
    "exp13-proportional":     "Proportional mix (10 / 90)",
}

# Fraction of TRAINING DATA that is promoter (rest is CDS). Drives the
# pie chart drawn beside each entry in the R2 line-plot legend, so the
# composition of each training condition reads at a glance.
PROMOTER_FRAC: dict[str, float] = {
    "exp21":              1.0,   # 100 % promoter
    "exp27":              0.0,   # 100 % CDS
    "exp13-equal":        0.5,   # uniform 50 / 50
    "exp13-proportional": 0.1,   # natural ~10 / 90 ratio (promoter / CDS)
}

# Two-panel subset set for the exp13 mixture sweep — one region from each
# end (promoter, the under-represented side; missense, a CDS subset).
# Labels match the R1 panel titles ("X variants") so the subplot title
# colours can be reused from REGION_TITLE_COLORS.
MIXTURE_SUBSETS: dict[str, str] = {
    "Promoter variants": "tss_proximal",
    "Missense variants": "missense_variant",
}

# Step coverage per exp13-mixture variant. exp21 / exp27 are the
# endpoints, exp13-equal / exp13-proportional are the mixtures.
MIXTURE_STEPS: dict[str, tuple[str, tuple[int, ...]]] = {
    # method_id → (S3 stem prefix, available steps on S3)
    "exp21":              ("exp21-promoters-yolo",       (2000, 6000, 10000, 12000, 14000, 16000, 18000, 20000, 22000)),
    "exp13-equal":        ("exp13-mixture-equal",        (2000, 6000, 10000, 14000, 18000, 22000, 26000)),
    "exp13-proportional": ("exp13-mixture-proportional", (2000, 6000, 10000, 14000, 18000, 22000, 26000)),
    "exp27":              ("exp27-cds-yolo",             (2000, 6000, 10000, 14000, 18000, 22000, 26000, 34000)),
}

# Three regional specialists × one matching variant consequence each,
# plus two "fair-scale" generalists (Evo 2 40B and GPN-Star M).
# Method order matches the cartoon order (enhancer / promoter / CDS),
# so the legend reads in the same direction as the gene cartoon above
# and the subplot row of the bar chart.
SPECIALIST_METHODS = ("exp136", "exp21", "exp27", "evo2_40b", "GPN-Star-M")

# Labels for the legend rows. The "Specialists" / "Generalists" group
# titles do the "-specialist" disambiguation, so the per-bar labels can
# stay short ("Promoter", "CDS", "Enhancer").
SPECIALIST_LABELS: dict[str, str] = {
    "exp21":      "Promoter",
    "exp27":      "CDS",
    "exp136":     "Enhancer",
    "evo2_40b":   "Evo 2 (40B)",
    "GPN-Star-M": "GPN-Star (M)",
}

# Per-method colour for this view. Specialists colour-coded to their
# trained region using the seaborn `colorblind` (Okabe-Ito-derived)
# palette so the three categorical regions are clearly distinguishable
# AND colour-blind safe. Same hexes used in the gene-cartoon blocks
# (figs/region_legend.svg) so the colour ties together: model legend,
# subplot title, and schematic block. Generalists kept as neutral GREYS
# so they read as "reference baselines, not region-coded".
SPECIALIST_COLORS: dict[str, str] = {
    "exp21":      "#0173b2",  # colorblind blue   — promoter
    "exp27":      "#de8f05",  # colorblind orange — CDS
    "exp136":     "#029e73",  # colorblind green  — enhancer
    "evo2_40b":   "#999999",  # mid grey  — baseline (Evo 2)
    "GPN-Star-M": "#555555",  # medium-dark grey — baseline (GPN-Star);
                              # was #333333 but the near-black bar made the
                              # SE error bar (ink) invisible against it.
}

# Three regions, one consequence each — the matching specialty. Order
# reads enhancer → promoter → missense (CDS) to match the gene cartoon's
# biological left-to-right layout and SPECIALIST_METHODS.
# Display labels carry "variants" so each subplot title reads as a
# noun phrase ("Enhancer variants") rather than a bare category.
SPECIALIST_REGIONS: dict[str, str] = {
    "Enhancer variants": "distal",
    "Promoter variants": "tss_proximal",
    "Missense variants": "missense_variant",
}

# Subplot title is coloured to match the corresponding specialist's bar
# (and the matching block in the gene-cartoon schematic). So the eye
# can link "Missense variants (orange)" → "CDS-specialist (orange)" →
# "CDS exons (orange)" without re-reading text.
REGION_TITLE_COLORS: dict[str, str] = {
    "Missense variants": "#de8f05",  # colorblind orange — CDS
    "Promoter variants": "#0173b2",  # colorblind blue   — promoter
    "Enhancer variants": "#029e73",  # colorblind green  — enhancer
}

# Final-step checkpoint per specialist (the same ones we use for R1/R2).
SPECIALIST_CHECKPOINTS: dict[str, str] = {
    "exp21":  "exp21-promoters-yolo-step-22000",
    "exp27":  "exp27-cds-yolo-step-34000",
    "exp136": "exp136-proj_v30-step-9999",
}


def _load_checkpoint(model_name: str) -> pl.DataFrame:
    """Load a single S3 mendelian_traits parquet for a given checkpoint."""
    uri = f"{S3_BASE}/{model_name}/mendelian_traits.parquet"
    return pl.read_parquet(uri)


def _specialist_grid() -> dict[str, dict[str, tuple[float, float]]]:
    """Assemble ``method_id → region_label → (auprc, se)``.

    Reads the final-step parquet for each of our 3 specialists, the Evo 2
    40B gist parquet, and the GPN-Star gist parquet (model = GPN-Star-M).
    Filters to the canonical score type per family.
    """
    grid: dict[str, dict[str, tuple[float, float]]] = {}

    # Specialists (marin_dna, mendelian_traits, minus_llr_avg)
    for sp, checkpoint in SPECIALIST_CHECKPOINTS.items():
        df = _load_checkpoint(checkpoint)
        per_region: dict[str, tuple[float, float]] = {}
        for label, subset in SPECIALIST_REGIONS.items():
            row = df.filter(
                (pl.col("score_type") == "minus_llr_avg")
                & (pl.col("subset") == subset)
            ).row(0, named=True)
            per_region[label] = (float(row["value"]), float(row["se"]))
        grid[sp] = per_region

    # Evo 2 40B
    evo = pl.read_parquet(
        f"{EVO2_METRICS_BASE}/mendelian_evo2_40b_train_metrics.parquet"
    ).filter(pl.col("score_type") == "minus_llr_avg")
    grid["evo2_40b"] = {
        label: tuple(
            evo.filter(pl.col("subset") == subset).select("value", "se").row(0)
        )
        for label, subset in SPECIALIST_REGIONS.items()
    }

    # GPN-Star M (calibrated cLLR, the paper's headline)
    gpn = pl.read_parquet(
        f"{GPN_STAR_METRICS_BASE}/mendelian_traits.GPN-Star.parquet"
    ).filter(
        (pl.col("score_type") == "minus_llr_calibrated")
        & (pl.col("model") == "GPN-Star-M")
    )
    grid["GPN-Star-M"] = {
        label: tuple(
            gpn.filter(pl.col("subset") == subset).select("value", "se").row(0)
        )
        for label, subset in SPECIALIST_REGIONS.items()
    }
    return grid


# (Removed: plot_specialist_radar and plot_specialist_profile were
# alternative views A/B-tested against the grouped bar chart below;
# the bar chart won, so the radar / profile functions are dropped.)


# Three-panel grouped bar chart: one panel per region, 5 bars each
# (3 specialists + 2 generalists). Per-panel y-axes (sharey=False)
# because the three regions are *different datasets* — bar heights are
# not directly comparable across panels (Missense AUPRC is intrinsically
# higher than Enhancer AUPRC because the datasets differ in difficulty).
# Single figure-level legend at the bottom; no xtick labels in any
# panel (colour carries the model identity).
def plot_specialist_grouped_bars(out_path: Path) -> None:
    apply_poster_style()
    grid = _specialist_grid()
    regions = list(SPECIALIST_REGIONS)
    xs = np.arange(len(SPECIALIST_METHODS))

    fig, axes = plt.subplots(
        1, len(regions),
        # Figsize matches the R3 line plot's figsize=(15, …) so the
        # POSTER_*_FS fonts render at the same physical size on both.
        # Height bumped proportionally from the original (11.5, 6.5)
        # so the panels stay roughly the same poster-vertical-space.
        figsize=(15, 8),
        sharey=False,        # each region is a different dataset
        constrained_layout=False,
    )

    # Plot, collecting bar handles separately so we can put specialists
    # and generalists on different legend rows below the panels.
    specialist_handles: list = []
    generalist_handles: list = []
    is_specialist = set(SPECIALIST_CHECKPOINTS)
    for ax_idx, (ax, region) in enumerate(zip(axes, regions)):
        heights = [grid[m][region][0] for m in SPECIALIST_METHODS]
        errs    = [grid[m][region][1] for m in SPECIALIST_METHODS]
        colors  = [SPECIALIST_COLORS[m] for m in SPECIALIST_METHODS]
        bars = ax.bar(
            xs, heights,
            color=colors,
            yerr=errs,
            edgecolor=OA_TEXT,
            linewidth=1.0,
            error_kw={"ecolor": OA_TEXT, "elinewidth": 3.0, "capsize": 0},
        )
        if ax_idx == 0:
            # Attach labels once and split into specialist / generalist groups.
            for bar, method in zip(bars, SPECIALIST_METHODS):
                bar.set_label(SPECIALIST_LABELS[method].replace("\n", " "))
                if method in is_specialist:
                    specialist_handles.append(bar)
                else:
                    generalist_handles.append(bar)

        # Per-bar value labels above the SE bar. Font kept small (15pt)
        # so adjacent annotations don't overlap when neighbouring bars
        # are similar height (e.g. Promoter 0.10 and CDS 0.09 in the
        # Enhancer panel — short bars in adjacent x positions whose
        # value labels would otherwise crash horizontally).
        for x, h, e in zip(xs, heights, errs):
            ax.text(
                x, h + e + 0.012,
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=15,
                color=OA_TEXT,
            )

        ax.set_title(
            region,
            fontsize=POSTER_TITLE_FS,
            fontweight="bold",
            pad=8,
            color=REGION_TITLE_COLORS[region],
        )
        # Drop the colour-encoded xtick clutter — model identity is in
        # the figure-level legend.
        ax.set_xticks([])
        # Independent y-axis per panel, but anchored at the matched-pair
        # chance baseline (AUPRC = 0.1 for 1:9 positive:negative ratio).
        ymax = max(h + e for h, e in zip(heights, errs)) + 0.08
        ax.set_ylim(0.1, ymax)
        ax.tick_params(axis="y", labelsize=POSTER_TICK_FS)
        # Force each panel's plotting box to be square so the three
        # subplots read as a uniform row of square cells (rather than
        # wide flat rectangles).
        ax.set_box_aspect(1.0)
        if ax_idx == 0:
            ax.set_ylabel("AUPRC", fontsize=POSTER_LABEL_FS)

    # Two figure-level legend rows below the three panels — one for the
    # 3 region specialists, one for the 2 whole-genome generalists.
    # The group label ("Specialists" / "Generalists") sits to the LEFT
    # of each row of entries (single row per group, no title-above-
    # entries gap to worry about).
    legend_kw = dict(
        loc="center left",
        fontsize=POSTER_LEGEND_FS,
        labelcolor=OA_TEXT,
        frameon=False,
        # Tight horizontal spacing so entries cluster together; defaults
        # (columnspacing=2.0, handletextpad=0.8) spread them too wide.
        columnspacing=0.6,
        handletextpad=0.3,
    )
    label_x = 0.06   # left edge of the group label
    legend_x = 0.32  # left edge of the legend, just past the longest label
    y_spec, y_gen = 0.34, 0.20

    fig.text(label_x, y_spec, "Specialists", ha="left", va="center",
             fontsize=POSTER_LEGEND_FS, fontweight="bold", color=OA_TEXT)
    fig.legend(
        handles=specialist_handles,
        bbox_to_anchor=(legend_x, y_spec),
        ncol=len(specialist_handles),
        **legend_kw,
    )

    fig.text(label_x, y_gen, "Generalists", ha="left", va="center",
             fontsize=POSTER_LEGEND_FS, fontweight="bold", color=OA_TEXT)
    fig.legend(
        handles=generalist_handles,
        bbox_to_anchor=(legend_x, y_gen),
        ncol=len(generalist_handles),
        **legend_kw,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.32, wspace=0.40)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


def _load_mixture_trajectory(method_id: str) -> pl.DataFrame:
    """Read all S3 checkpoints for one exp13-sweep variant."""
    stem, steps = MIXTURE_STEPS[method_id]
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    for step in steps:
        uri = f"{S3_BASE}/{stem}-step-{step}/mendelian_traits.parquet"
        try:
            ck = pl.read_parquet(uri)
        except Exception as exc:
            missing.append(f"  {method_id} step {step}: {exc}")
            continue
        parts.append(ck.with_columns(pl.lit(step).alias("step")))
    if missing:
        print(
            f"WARN: {len(missing)}/{len(steps)} {method_id} parquets unread:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, f"no {method_id} parquets loaded"
    return pl.concat(parts)


class HandlerLineMarkerWithPie(HandlerBase):
    """Native-legend handler: line + central marker + two-tone pie.

    Renders the legend handle as `[line + marker | pie]` (left to right),
    with the pie encoding a 2-component ratio for the method represented
    by this handle. Because it subclasses HandlerBase, the line and
    marker are drawn by matplotlib's own legend renderer using the
    proxy artist's actual stroke width / marker size — guaranteed to
    match the line plot's lines pixel-for-pixel.

    One handler instance is constructed per method (each carries its own
    ``promoter_frac``); the per-handle handler_map is passed to
    ``fig.legend(handler_map=...)``.
    """

    def __init__(
        self,
        promoter_frac: float,
        promoter_color: str,
        cds_color: str,
        edge_color: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.promoter_frac = promoter_frac
        self.promoter_color = promoter_color
        self.cds_color = cds_color
        self.edge_color = edge_color

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans,
    ):
        # Reserve the right portion of the handle box for the pie; the
        # line + marker share the rest. Pie diameter = full handle height
        # so it visually balances the line's thickness × marker.
        pie_d = height
        gap = 0.35 * fontsize   # spacing between line-end and pie
        line_y = height / 2 - ydescent
        line_x0 = -xdescent
        line_x1 = -xdescent + width - pie_d - gap

        color = orig_handle.get_color()
        line = Line2D(
            [line_x0, line_x1], [line_y, line_y],
            color=color, lw=orig_handle.get_linewidth(),
            solid_capstyle="butt",
        )
        marker = Line2D(
            [(line_x0 + line_x1) / 2], [line_y],
            color=color,
            marker=orig_handle.get_marker(),
            markersize=orig_handle.get_markersize(),
            markeredgecolor=self.edge_color,
            markeredgewidth=1.0,
            linestyle="None",
        )
        line.set_transform(trans)
        marker.set_transform(trans)
        artists: list = [line, marker]

        cx = -xdescent + width - pie_d / 2
        cy = line_y
        # Slight inset so the wedge edge stroke isn't clipped by the handle box.
        r = pie_d / 2 * 0.92
        p = self.promoter_frac
        if p >= 1.0:
            c = Circle(
                (cx, cy), r,
                facecolor=self.promoter_color, edgecolor=self.edge_color, linewidth=1.0,
            )
            c.set_transform(trans)
            artists.append(c)
        elif p <= 0.0:
            c = Circle(
                (cx, cy), r,
                facecolor=self.cds_color, edgecolor=self.edge_color, linewidth=1.0,
            )
            c.set_transform(trans)
            artists.append(c)
        else:
            # Promoter slice starts at 12 o'clock, grows clockwise — so
            # the small 10% slice for proportional sits at the top-right
            # and the eye reads ratio at a glance. matplotlib's Wedge is
            # CCW from theta1 to theta2; we set theta1 below 90 to span
            # the clockwise arc.
            theta_top = 90.0
            theta_start = theta_top - p * 360.0
            w_p = Wedge(
                (cx, cy), r, theta_start, theta_top,
                facecolor=self.promoter_color, edgecolor=self.edge_color, linewidth=1.0,
            )
            w_c = Wedge(
                (cx, cy), r, theta_top, theta_top + (1.0 - p) * 360.0,
                facecolor=self.cds_color, edgecolor=self.edge_color, linewidth=1.0,
            )
            w_p.set_transform(trans)
            w_c.set_transform(trans)
            artists.append(w_p)
            artists.append(w_c)
        return artists


# R3: exp13 promoter × CDS mixture sweep — equal-compute trajectories.
# The 4 variants have different "final" step counts, so a final-step bar
# chart would compare them at unequal compute. Plotting AUPRC vs training
# step makes the equal-compute comparison visible at every vertical slice.
def plot_r3(out_path: Path) -> None:
    """4 mixture variants (100%P / 50-50 / 10-90 / 100%C) over training.

    Three panels (Promoter / Missense / Average) so the trade-off shows
    up: 100% P does well on promoter and badly on missense; 100% C the
    mirror image; 50/50 lands well on both; 10/90 (proportional / naive)
    ignores promoters and tracks 100% C.

    A native fig.legend() sits below all three panels with a custom
    handler (`HandlerLineMarkerWithPie`) that draws the same line +
    marker matplotlib uses for the data lines, alongside a small pie
    chart encoding the method's promoter / CDS training-data ratio.
    """
    apply_poster_style()
    methods = ("exp21", "exp13-equal", "exp13-proportional", "exp27")
    score_type = "minus_llr_avg"

    # Pre-load all 4 trajectories.
    traj = {m: _load_mixture_trajectory(m) for m in methods}

    # Only plot training steps that are available in ALL four runs —
    # apples-to-apples comparison at every x-tick. Intersection over
    # MIXTURE_STEPS gives {2000, 6000, 10000, 14000, 18000, 22000}
    # at time of writing; expand if more shared checkpoints become
    # available upstream.
    common_steps = set.intersection(
        *[set(MIXTURE_STEPS[m][1]) for m in methods]
    )

    # Fonts: use the module-level POSTER_*_FS constants so this line
    # plot and the R1 specialist bar chart stay in sync at the per-
    # element level.
    title_fs  = POSTER_TITLE_FS
    label_fs  = POSTER_LABEL_FS
    tick_fs   = POSTER_TICK_FS
    legend_fs = POSTER_LEGEND_FS

    # Three panels: Promoter | Missense | Average (mean across both).
    # The average panel makes the headline "Uniform mix wins on aggregate"
    # visible directly — in the two-region panels you have to trade off
    # mentally; here you just look at which line is highest. Extra height
    # (4.5 → 6.5) makes room for the 2-row native fig.legend below.
    fig, axes = plt.subplots(
        1, len(MIXTURE_SUBSETS) + 1, figsize=(15, 6.5), sharey=False,
    )
    line_handles: dict[str, Line2D] = {}
    for ax, (panel, subset_key) in zip(axes[:-1], MIXTURE_SUBSETS.items()):
        for m in methods:
            sub = (
                traj[m]
                .filter(
                    (pl.col("score_type") == score_type)
                    & (pl.col("subset") == subset_key)
                    & (pl.col("step").is_in(list(common_steps)))
                )
                .sort("step")
            )
            (line,) = ax.plot(
                sub["step"].to_numpy(),
                sub["value"].to_numpy(),
                marker="o",
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
            )
            line_handles[m] = line  # any panel's Line2D is fine for the legend
        ax.set_xlabel("training step", fontsize=label_fs)
        if ax is axes[0]:
            ax.set_ylabel("AUPRC", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        # Colour the subplot title to match the corresponding region
        # (matches R1 + the gene cartoon above) so the eye ties the
        # panel to the right block in the schematic.
        ax.set_title(
            panel,
            fontweight="bold",
            fontsize=title_fs,
            color=REGION_TITLE_COLORS[panel],
        )

    # Third panel: per-method average of (promoter, missense) at each
    # shared step.
    ax_avg = axes[-1]
    for m in methods:
        sub = (
            traj[m]
            .filter(
                (pl.col("score_type") == score_type)
                & (pl.col("subset").is_in(list(MIXTURE_SUBSETS.values())))
                & (pl.col("step").is_in(list(common_steps)))
            )
            .group_by("step")
            .agg(pl.col("value").mean().alias("value"))
            .sort("step")
        )
        ax_avg.plot(
            sub["step"].to_numpy(),
            sub["value"].to_numpy(),
            marker="o",
            color=METHOD_COLORS[m],
            label=METHOD_LABELS[m],
        )
    ax_avg.set_xlabel("training step", fontsize=label_fs)
    ax_avg.tick_params(axis="both", labelsize=tick_fs)
    ax_avg.set_title(
        "Average",
        fontweight="bold",
        fontsize=title_fs,
        color=OA_TEXT,
    )

    # X-ticks: let matplotlib's auto-locator choose so they're uniformly
    # spaced. (Manual subsetting picked first / mid / last which gave
    # non-uniform gaps — looked weird.)

    # ── Native fig.legend, 2×2, custom handler per method ──
    # ncol=2 (not 4) so the larger font fits without overflowing the
    # figure width. matplotlib fills the legend column-major, so the
    # `legend_order` tuple below is constructed so the 2×2 grid reads:
    #
    #     Promoter only          CDS only        (pure datasets, top row)
    #     Uniform mix            Proportional mix (mixtures, bottom row)
    legend_order = ("exp21", "exp13-equal", "exp27", "exp13-proportional")
    handler_map = {
        line_handles[m]: HandlerLineMarkerWithPie(
            promoter_frac=PROMOTER_FRAC[m],
            promoter_color="#0173b2",   # mirrors gene cartoon's promoter blue
            cds_color="#de8f05",        # mirrors gene cartoon's CDS orange
            edge_color=OA_TEXT,
        )
        for m in methods
    }
    fig.legend(
        [line_handles[m] for m in legend_order],
        [METHOD_LABELS[m]  for m in legend_order],
        handler_map=handler_map,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=legend_fs,
        handlelength=4.0,      # room for line + marker + pie inside the handle
        handletextpad=0.4,
        columnspacing=2.0,
        labelspacing=0.6,
    )
    # Leave room at the bottom for the 2-row legend; widen the left
    # margin so the (now larger) AUPRC ylabel + ticks aren't clipped.
    fig.subplots_adjust(bottom=0.42, top=0.93, left=0.09, right=0.98, wspace=0.28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


# ─── Entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    apply_poster_style()
    plot_timescale(FIGS_DIR / "timescale.svg")
    plot_r3(FIGS_DIR / "r3.svg")
    plot_specialist_grouped_bars(FIGS_DIR / "specialist_bars.svg")
