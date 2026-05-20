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

# Per-method colour. Our hero models (exp21, exp27) = OA copper. Evo 2
# in a cool blue/teal family. GPN-Star in plum.
#
# For the exp13 mixture sweep, the 4 entries (100%P, 50/50, 10/90, 100%C)
# are anchored on the OA region colours at the extremes (copper = promoter,
# brick = CDS) with two OA colorway hops in between, so each mixture has
# a distinct identity AND the endpoints visually tie back to the region
# legend.
METHOD_COLORS: dict[str, str] = {
    "exp21":                  "#9e6d43",  # OA copper (100% promoter)
    "exp27":                  "#7a3b2e",  # OA brick (100% CDS)
    "exp13-equal":            "#a86a2c",  # OA burnt orange (50/50 mixture)
    "exp13-proportional":     "#6b5b3e",  # OA olive (10/90 mixture, biased to CDS)
    "evo2_1b":                "#7BAFC4",  # light teal
    "evo2_7b":                "#3D7A92",  # medium teal
    "evo2_40b":               "#1F4A5A",  # dark teal
    "GPN-Star-V":             "#C68DAC",  # light plum
    "GPN-Star-M":             "#8B3A62",  # OA plum
    "GPN-Star-P":             "#5A1F3F",  # dark plum
}
METHOD_LABELS: dict[str, str] = {
    "exp21":                  "exp21 (100% promoter)",
    "exp27":                  "exp27 (100% CDS)",
    "exp13-equal":            "exp13 (50 / 50 mix)",
    "exp13-proportional":     "exp13 (10 / 90 mix)",
    "evo2_1b":                "Evo 2 (1B)",
    "evo2_7b":                "Evo 2 (7B)",
    "evo2_40b":               "Evo 2 (40B)",
    "GPN-Star-V":             "GPN-Star (V)",
    "GPN-Star-M":             "GPN-Star (M)",
    "GPN-Star-P":             "GPN-Star (P)",
}

# Subsets we plot side-by-side. exp21 was trained on promoters, so the
# 5'UTR panel doubles as an off-target generalisation check.
R1_SUBSETS: dict[str, str] = {
    "Promoter":  "tss_proximal",
    "5' UTR":    "5_prime_UTR_variant",
}

# CDS-related subsets for R2 (exp27 was trained on CDS).
R2_SUBSETS: dict[str, str] = {
    "Missense":   "missense_variant",
    "Splicing":   "splicing",
    "Synonymous": "synonymous_variant",
}

# Two-panel subset set for the exp13 mixture sweep — one region from each
# end (promoter, the under-represented side; missense, a CDS subset).
MIXTURE_SUBSETS: dict[str, str] = {
    "Promoter":  "tss_proximal",
    "Missense":  "missense_variant",
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
SPECIALIST_METHODS = ("exp21", "exp27", "exp136", "evo2_40b", "GPN-Star-M")

SPECIALIST_LABELS: dict[str, str] = {
    "exp21":      "exp21\n(promoter)",
    "exp27":      "exp27\n(CDS)",
    "exp136":     "exp136\n(enhancer)",
    "evo2_40b":   "Evo 2 (40B)",
    "GPN-Star-M": "GPN-Star (M)",
}

# Per-method colour for this view. Specialists colour-coded to their
# trained region (matches the gene-cartoon legend); generalists in their
# family colours (teal for Evo 2, plum for GPN-Star).
SPECIALIST_COLORS: dict[str, str] = {
    "exp21":      "#9e6d43",  # OA copper — promoter
    "exp27":      "#7a3b2e",  # OA brick — CDS
    "exp136":     "#6b5b3e",  # OA olive — enhancer
    "evo2_40b":   "#1F4A5A",  # dark teal
    "GPN-Star-M": "#8B3A62",  # OA plum
}

# Three regions, one consequence each — the matching specialty.
SPECIALIST_REGIONS: dict[str, str] = {
    "Promoter": "tss_proximal",
    "Missense": "missense_variant",
    "Enhancer": "distal",
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


def _final_value(model_name: str, subset: str, score_type: str = "minus_llr_avg"
                 ) -> tuple[float, float]:
    """Return ``(value, se)`` for ``model_name`` on ``subset``."""
    df = _load_checkpoint(model_name).filter(
        (pl.col("score_type") == score_type)
        & (pl.col("subset") == subset)
    )
    row = df.row(0, named=True)
    return (float(row["value"]), float(row["se"]))


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


def _plot_comparison_bars(
    hero_id: str,
    hero_checkpoint: str,
    subsets: dict[str, str],
    out_path: Path,
) -> None:
    """Shared body for R1 / R2: per-subset bar chart of one hero model
    (exp21 or exp27) against the same Evo 2 + GPN-Star baselines, with
    per-cluster bootstrap SE error bars."""
    methods = [hero_id, "evo2_1b", "evo2_7b", "evo2_40b",
               "GPN-Star-V", "GPN-Star-M", "GPN-Star-P"]
    n_panels = len(subsets)
    fig_width = 6 + 2.8 * n_panels  # roughly 11.5 for 2 panels, 14 for 3
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (panel, subset_key) in zip(axes, subsets.items()):
        hero_value = _final_value(hero_checkpoint, subset_key)
        values = {hero_id: hero_value, **_baseline_values(subset_key)}

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
            # SE bars without caps — cleaner read at poster distance.
            error_kw={"ecolor": OA_TEXT, "elinewidth": 1.2, "capsize": 0},
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
        for x, h, e in zip(xs, heights, errs):
            ax.text(x, h + e + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


# R1: exp21 (promoter-trained) on the two regions it was scored against.
def plot_r1(out_path: Path) -> None:
    """exp21 vs Evo 2 / GPN-Star on Promoter and 5'UTR."""
    _plot_comparison_bars(
        hero_id="exp21",
        hero_checkpoint="exp21-promoters-yolo-step-22000",
        subsets=R1_SUBSETS,
        out_path=out_path,
    )


# R2: exp27 (CDS-trained) on the three CDS variant subsets.
def plot_r2(out_path: Path) -> None:
    """exp27 vs Evo 2 / GPN-Star on Missense, Splicing, and Synonymous."""
    _plot_comparison_bars(
        hero_id="exp27",
        hero_checkpoint="exp27-cds-yolo-step-34000",
        subsets=R2_SUBSETS,
        out_path=out_path,
    )


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


# R-radar: 5 methods (3 specialists + 2 generalists) on 3 regions —
# poster-friendly single-image takeaway. Each specialist's spike lands
# on its own training region.
def plot_specialist_radar(out_path: Path) -> None:
    grid = _specialist_grid()
    regions = list(SPECIALIST_REGIONS)
    n = len(regions)

    # Angles for each axis — start at the top (π/2) and go counter-clockwise
    # for the "spike at top" feel of GPN-Star Fig 1B-style radars.
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    # Append the first angle to close the polygon.
    closed_angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7.5, 7.0), subplot_kw={"projection": "polar"})
    for method in SPECIALIST_METHODS:
        vals = [grid[method][r][0] for r in regions]
        closed_vals = vals + [vals[0]]
        is_specialist = method in SPECIALIST_CHECKPOINTS
        ax.plot(
            closed_angles, closed_vals,
            color=SPECIALIST_COLORS[method],
            linewidth=2.5 if is_specialist else 2.0,
            label=SPECIALIST_LABELS[method].replace("\n", " "),
            zorder=5 if is_specialist else 3,
        )

    # Radial scale: 0 to 0.7 covers all data + a touch of headroom.
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=10, color=OA_TEXT_LIGHT)

    # Region labels at each spoke
    ax.set_xticks(angles)
    ax.set_xticklabels(regions, fontsize=14, fontweight="bold", color=OA_TEXT)

    # Move the radial labels off the axis line so they don't sit on top
    # of the polygons.
    ax.set_rlabel_position(105)
    ax.spines["polar"].set_color(OA_TEXT_LIGHT)
    ax.grid(color=OA_TEXT_LIGHT, alpha=0.4, linewidth=0.7)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        labelcolor=OA_TEXT,
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


# Alternative view: 3 small bar charts, one per region, 5 bars each.
# Same data, less interpretive risk, slightly less striking. Side-by-
# side fallback to the radar.
def plot_specialist_grouped_bars(out_path: Path) -> None:
    grid = _specialist_grid()
    regions = list(SPECIALIST_REGIONS)

    fig, axes = plt.subplots(1, len(regions), figsize=(11.5, 4.0), sharey=True)
    for ax, region in zip(axes, regions):
        heights = [grid[m][region][0] for m in SPECIALIST_METHODS]
        errs    = [grid[m][region][1] for m in SPECIALIST_METHODS]
        colors  = [SPECIALIST_COLORS[m] for m in SPECIALIST_METHODS]
        xs      = list(range(len(SPECIALIST_METHODS)))
        ax.bar(
            xs, heights,
            yerr=errs,
            color=colors,
            edgecolor=OA_TEXT,
            linewidth=1.0,
            error_kw={"ecolor": OA_TEXT, "elinewidth": 1.2, "capsize": 0},
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [SPECIALIST_LABELS[m].replace("\n", " ") for m in SPECIALIST_METHODS],
            rotation=35, ha="right", fontsize=10,
        )
        ax.set_title(region)
        if ax is axes[0]:
            ax.set_ylabel("AUPRC")
        # AUPRC=0.1 is the matched-pair chance baseline (1:9 positive:negative
        # ratio), so start the y-axis there — bars now show signal above chance.
        ax.set_ylim(bottom=0.1)
        for x, h, e in zip(xs, heights, errs):
            ax.text(x, h + e + 0.012, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

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


# R3: exp13 promoter × CDS mixture sweep — equal-compute trajectories.
# The 4 variants have different "final" step counts, so a final-step bar
# chart would compare them at unequal compute. Plotting AUPRC vs training
# step makes the equal-compute comparison visible at every vertical slice.
def plot_r3(out_path: Path) -> None:
    """4 mixture variants (100%P / 50-50 / 10-90 / 100%C) over training.

    Two panels (Promoter / Missense) — one region from each side of the
    sweep — so the trade-off shows up: 100% P does well on promoter and
    badly on missense; 100% C the mirror image; 50/50 lands well on both;
    10/90 (proportional / naive) ignores promoters and tracks 100% C.
    """
    methods = ("exp21", "exp13-equal", "exp13-proportional", "exp27")
    score_type = "minus_llr_avg"

    # Pre-load all 4 trajectories.
    traj = {m: _load_mixture_trajectory(m) for m in methods}

    fig, axes = plt.subplots(1, len(MIXTURE_SUBSETS), figsize=(11.5, 4.5), sharey=False)
    for ax, (panel, subset_key) in zip(axes, MIXTURE_SUBSETS.items()):
        for m in methods:
            sub = (
                traj[m]
                .filter(
                    (pl.col("score_type") == score_type)
                    & (pl.col("subset") == subset_key)
                )
                .sort("step")
            )
            ax.plot(
                sub["step"].to_numpy(),
                sub["value"].to_numpy(),
                marker="o",
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
            )
        ax.set_xlabel("training step")
        if ax is axes[0]:
            ax.set_ylabel("AUPRC")
        ax.set_title(panel)

    # Single legend to the right — the 4 methods are common across panels.
    axes[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title=None,
        labelcolor=OA_TEXT,
    )

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
    plot_r2(FIGS_DIR / "r2.svg")
    plot_r3(FIGS_DIR / "r3.svg")
    plot_specialist_radar(FIGS_DIR / "specialist_radar.svg")
    plot_specialist_grouped_bars(FIGS_DIR / "specialist_bars.svg")
