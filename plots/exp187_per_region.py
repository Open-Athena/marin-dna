"""Plots for exp187 per-region Qwen3-1B sweep (issue #187).

**Metric**: AUPRC + cluster-bootstrap SE (PR #195 schema). Earlier iterations
of this script used PairwiseAccuracy on the 1:1 matched-pair dataset from
#186; PR #194 rebuilt the mendelian dataset at k=9 (1:9 pos:neg ratio) where
PA is no longer valid, so AUPRC took over. Random-classifier baseline on a
1:9 dataset is **0.10** (= prevalence), not 0.5.

Outputs (all under plots/output/, gitignored). Every figure is saved as both
`.png` (130 dpi for inline preview) and `.svg` (vector, for issue embeds):

  exp187_auprc_training_curves_llr.{png,svg}
  exp187_auprc_training_curves_jsd.{png,svg}
  exp187_auprc_training_curves_llr_noerr.{png,svg}
  exp187_auprc_training_curves_jsd_noerr.{png,svg}
      One panel per **real** mendelian subset (`_global_` and `_macro_avg_`
      sentinels are omitted by request). One line per training-region arm,
      thicker line for the plan's predicted diagonal winner. Two score_types
      × {with-errorbars, no-errorbars} = 4 figures.

  exp187_auprc_heatmap_llr.{png,svg}
  exp187_auprc_heatmap_jsd.{png,svg}
      6×8 heatmap of final-checkpoint (step-4999) AUPRC: rows = arms (plan
      order), columns = subsets (ordered to make the block-diagonal of
      plan-expected winners visually obvious); diagonal cells boxed; green
      star = actual column-max.

  exp187_train_loss.{png,svg}
      Training loss curves pulled from WandB. Legend labels include the
      number of full passes through each arm's HF dataset reached at the
      5K-step budget — small-data arms epoch many times, big-data arms
      stay under one epoch.

  exp187_llgap_training_curves.{png,svg}
      One panel per `val_*` recipe; line per arm × in-training eval step.
      Cheap proxy for what AUPRC will look like; flattening / dipping LL
      gap on a recipe = the arm has stopped learning subset structure.

  exp187_llgap_vs_auprc.{png,svg}
      Scatter of in-training LL gap (WandB) vs offline AUPRC per (val_*,
      subset). Sigmoid fit + correlation annotations (issue #8 convention).

Run:
    uv run python plots/exp187_per_region.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.patches import Rectangle
from scipy import optimize, stats

import wandb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARMS: list[str] = [
    "v3_cds",
    "v3_utr3",
    "v3_ncrna_exon",
    "v3_tss_region_and_utr5",
    "v3_ccre_non_promoter",
    "v3_bg",
]

# 4 intermediate HF-exported checkpoints (this PR) + step-4999 final.
STEPS: list[int] = [1000, 2000, 3000, 4000, 4999]
FINAL_STEP: int = 4999

# 5 LL-gap val recipes used in the in-training eval (PR #171).
VAL_RECIPES: list[str] = [
    "val_cds",
    "val_utr3",
    "val_ncrna",
    "val_tss_pc",
    "val_enhancer",
]

# Per-recipe → list-of-mendelian-subset(s) mapping for the scatter panels.
RECIPE_TO_SUBSETS: dict[str, list[str]] = {
    "val_cds": ["missense_variant", "synonymous_variant", "splicing"],
    "val_utr3": ["3_prime_UTR_variant"],
    "val_ncrna": ["non_coding_transcript_exon_variant"],
    "val_tss_pc": ["5_prime_UTR_variant", "tss_proximal"],
    "val_enhancer": ["distal"],
}

# Diagonal-wins map from the plan (arm → its expected-winner subsets).
DIAGONAL: dict[str, set[str]] = {
    "v3_cds": {"missense_variant", "synonymous_variant", "splicing"},
    "v3_utr3": {"3_prime_UTR_variant"},
    "v3_ncrna_exon": {"non_coding_transcript_exon_variant"},
    "v3_tss_region_and_utr5": {"5_prime_UTR_variant", "tss_proximal"},
    "v3_ccre_non_promoter": {"distal"},
    "v3_bg": set(),
}

# Subset order for heatmaps + PA-curve panels — groups predicted winners by
# arm so the block-diagonal of expected wins reads top-left → bottom-right.
SUBSET_ORDER: list[str] = [
    # v3_cds territory
    "missense_variant",
    "synonymous_variant",
    "splicing",
    # v3_utr3 territory
    "3_prime_UTR_variant",
    # v3_ncrna_exon territory
    "non_coding_transcript_exon_variant",
    # v3_tss_region_and_utr5 territory
    "5_prime_UTR_variant",
    "tss_proximal",
    # v3_ccre_non_promoter territory
    "distal",
]
# Sentinels we explicitly exclude per user request — no `_global_` /
# `_macro_avg_` in this analysis.
SENTINELS: set[str] = {"_global_", "_macro_avg_"}

# Okabe-Ito colorblind-safe palette — one color per arm.
ARM_COLORS: dict[str, str] = {
    "v3_cds": "#E69F00",  # orange
    "v3_utr3": "#56B4E9",  # sky blue
    "v3_ncrna_exon": "#009E73",  # bluish green
    "v3_tss_region_and_utr5": "#F0E442",  # yellow
    "v3_ccre_non_promoter": "#0072B2",  # blue
    "v3_bg": "#D55E00",  # vermillion
}

# Epochs of training each arm reaches at 5K steps × batch 8192 × 256 tokens
# = ~10.5e9 tokens-equivalent. From the plan body's table (post-RC row counts
# of the v3_<arm> HF datasets). Used only as a legend annotation on the train-
# loss plot so the reader sees at a glance which arms epoched many times (and
# therefore had repeated exposures to the same windows, the overfitting risk).
ARM_EPOCHS: dict[str, float] = {
    "v3_ccre_non_promoter": 0.42,  # 96.6M rows
    "v3_cds": 0.52,  # 78.4M rows
    "v3_ncrna_exon": 2.68,  # 15.3M rows
    "v3_bg": 2.70,  # 15.1M rows
    "v3_utr3": 3.98,  # 10.3M rows
    "v3_tss_region_and_utr5": 5.04,  # 8.1M rows
}

# PR #195 emits per-strand atoms and 3 strand-aggregations per dataset's
# score protocol — we headline on the `_avg` strand (FWD+RC averaged, then
# protocol-transformed). Mendelian dataset uses `minus_llr` protocol per
# config.yaml; JSD doesn't sign-flip so keeps its raw name.
SCORE_TYPES: list[tuple[str, str]] = [
    ("minus_llr_avg", "LLR"),
    ("jsd_avg", "JSD"),
]

# AUPRC random-classifier baseline = positive prevalence. The PR #194 k=9
# dataset has a strict 1:9 pos:neg ratio per (chrom, consequence_final) stratum,
# so per-subset baselines are 0.10. `_global_` and `_macro_avg_` are also at
# 0.10 by construction. Used for the dotted chance line on each panel.
AUPRC_BASELINE: float = 0.10

S3_PREFIX: str = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
WANDB_PROJECT: str = "marin"
CACHE_DIR: Path = Path(__file__).parent / "output" / ".wandb_cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure as both PNG (130 dpi) and SVG (vector). Close the fig."""
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png.name} + {svg.name}")


def _sigmoid(
    x: np.ndarray, lower: float, upper: float, k: float, x0: float
) -> np.ndarray:
    """Increasing sigmoid: lower at low x, upper at high x."""
    return lower + (upper - lower) / (1 + np.exp(np.clip(-k * (x - x0), -500, 500)))


def _fit_sigmoid(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Fit a sigmoid to x vs y; return popt or None if fit fails."""
    try:
        k0 = 5.0 if np.corrcoef(x, y)[0, 1] >= 0 else -5.0
        p0 = [y.min(), y.max(), k0, float(np.median(x))]
        bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
        popt, _ = optimize.curve_fit(_sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except (RuntimeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_offline_pa() -> pl.DataFrame:
    """Load the 30 (arm × step) AUPRC metric parquets from S3.

    Missing parquets are tolerated (warning printed) so plots can be
    regenerated mid-sweep — affected cells render as gaps rather than
    crashing the whole script.
    """
    frames: list[pl.DataFrame] = []
    missing: list[tuple[str, int]] = []
    for arm in ARMS:
        for step in STEPS:
            uri = f"{S3_PREFIX}/exp187-{arm}-step-{step}/mendelian_traits.parquet"
            try:
                df = pl.read_parquet(uri).with_columns(
                    pl.lit(arm).alias("arm"),
                    pl.lit(step).alias("step"),
                )
            except OSError as e:
                if "404 Not Found" in str(e):
                    missing.append((arm, step))
                    continue
                raise
            frames.append(df)
    if missing:
        print(
            f"  WARNING: {len(missing)} parquet(s) missing on S3 — rendering with gaps:"
        )
        for arm, step in missing:
            print(f"    - exp187-{arm}-step-{step}")
    return pl.concat(frames, how="vertical")


def _run_for_arm(api: wandb.Api, arm: str) -> "wandb.apis.public.Run":
    runs = api.runs(
        WANDB_PROJECT,
        filters={
            "tags": {"$all": ["exp187", f"region={arm}"]},
            "config.trainer.num_train_steps": 5000,
        },
    )
    candidates = [r for r in runs if "smoke" not in r.tags and r.state == "finished"]
    assert candidates, f"no finished prod run for {arm}"
    candidates.sort(key=lambda r: r.created_at, reverse=True)
    return candidates[0]


def _step_col(hist: pd.DataFrame) -> pd.Series:
    if "global_step" in hist.columns and hist["global_step"].notna().any():
        s = hist["global_step"].fillna(hist["_step"])
    else:
        s = hist["_step"]
    return s.astype(int)


def load_wandb_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train_df, eval_df) with `arm` + `step` columns. Cached to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_cache = CACHE_DIR / "exp187_train_loss.parquet"
    eval_cache = CACHE_DIR / "exp187_eval_llgap.parquet"

    if train_cache.exists() and eval_cache.exists():
        print(f"  Using cached wandb history at {CACHE_DIR}/")
        return pd.read_parquet(train_cache), pd.read_parquet(eval_cache)

    api = wandb.Api()
    eval_keys = [
        f"eval/{r}_{kind}/loss"
        for r in VAL_RECIPES
        for kind in ("functional", "nonfunctional")
    ]

    train_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []
    for arm in ARMS:
        print(f"  fetching wandb history for {arm} ...")
        run = _run_for_arm(api, arm)

        t = run.history(keys=["train/loss"], samples=10000, pandas=True)
        t["arm"] = arm
        t["step"] = _step_col(t)
        train_frames.append(t[["arm", "step", "train/loss"]])

        e = run.history(keys=eval_keys, samples=10000, pandas=True)
        e["arm"] = arm
        e["step"] = _step_col(e)
        for r in VAL_RECIPES:
            f_col = f"eval/{r}_functional/loss"
            n_col = f"eval/{r}_nonfunctional/loss"
            e[f"{r}_llgap"] = e[n_col] - e[f_col]
        eval_frames.append(e[["arm", "step"] + [f"{r}_llgap" for r in VAL_RECIPES]])

    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)
    train_df.to_parquet(train_cache)
    eval_df.to_parquet(eval_cache)
    print(f"  Cached: train_df={len(train_df)} rows, eval_df={len(eval_df)} rows")
    return train_df, eval_df


# ---------------------------------------------------------------------------
# Plot 1: AUPRC training curves (with + without errorbars × LLR + JSD)
# ---------------------------------------------------------------------------


def plot_auprc_curves(
    df: pl.DataFrame,
    score_type: str,
    label: str,
    out_dir: Path,
    *,
    with_errorbars: bool,
) -> None:
    """One panel per real subset (no `_global_` / `_macro_avg_`)."""
    sub = df.filter(
        (pl.col("score_type") == score_type) & (~pl.col("subset").is_in(SENTINELS))
    )
    panel_order = [s for s in SUBSET_ORDER if s in sub["subset"].unique().to_list()]

    n_panels = len(panel_order)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    for i, subset in enumerate(panel_order):
        ax = axes[i // n_cols][i % n_cols]
        sub_panel = sub.filter(pl.col("subset") == subset).sort(["arm", "step"])
        for arm in ARMS:
            arm_df = sub_panel.filter(pl.col("arm") == arm).sort("step")
            if arm_df.is_empty():
                continue
            xs = arm_df["step"].to_list()
            ys = arm_df["value"].to_list()
            es = arm_df["se"].to_list()
            highlight = subset in DIAGONAL[arm]
            common = dict(
                marker="o",
                markersize=5 if highlight else 4,
                linewidth=2.5 if highlight else 1.2,
                color=ARM_COLORS[arm],
                label=arm,
                alpha=0.95 if highlight else 0.75,
            )
            if with_errorbars:
                ax.errorbar(xs, ys, yerr=es, capsize=2, **common)
            else:
                ax.plot(xs, ys, **common)
        ax.axhline(AUPRC_BASELINE, linestyle=":", color="gray", linewidth=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel("AUPRC")
        ax.set_title(subset, fontsize=10)
        ax.grid(True, alpha=0.3)

    for j in range(n_panels, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    handles, labels_l = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_l,
        loc="lower center",
        ncol=len(ARMS),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )
    err_suffix = " with SE bars" if with_errorbars else " (no error bars)"
    fig.suptitle(
        f"exp187 — Mendelian AUPRC training curves ({label}){err_suffix}\n"
        f"thick line = expected diagonal winner per plan; "
        f"dotted = chance (1:9 baseline = {AUPRC_BASELINE:.2f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    stem = f"exp187_auprc_training_curves_{label.lower()}"
    if not with_errorbars:
        stem += "_noerr"
    _savefig(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Plot 2: PA heatmap (6 arms × 8 real subsets) at final step
# ---------------------------------------------------------------------------


def plot_auprc_heatmap(
    df: pl.DataFrame, score_type: str, label: str, out_dir: Path
) -> None:
    """6 × 8 heatmap of AUPRC at step-4999. Rows = arms (plan order), cols =
    subsets (SUBSET_ORDER, grouped so block-diagonal of plan-expected
    winners is visible). Diagonal cells get a thick black border."""
    sub = (
        df.filter(
            (pl.col("score_type") == score_type)
            & (pl.col("step") == FINAL_STEP)
            & (~pl.col("subset").is_in(SENTINELS))
        )
        .to_pandas()
        .set_index(["arm", "subset"])["value"]
        .unstack("subset")
    )
    matrix = sub.loc[ARMS, SUBSET_ORDER]

    fig, ax = plt.subplots(figsize=(1.05 * len(SUBSET_ORDER) + 2, 0.7 * len(ARMS) + 2))
    # `Reds` (sequential, no diverging center) — AUPRC has a one-sided "good"
    # direction (higher = better) and a hard prevalence floor at ~0.10, so a
    # diverging palette centered on 0.5 wastes contrast on values that don't
    # appear. vmax=0.6 caps the brightest cells so per-subset spread reads
    # cleanly; nothing in this experiment is above that.
    im = ax.imshow(
        matrix.values,
        cmap="Reds",
        vmin=AUPRC_BASELINE,
        vmax=0.6,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(SUBSET_ORDER)))
    ax.set_xticklabels(SUBSET_ORDER, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(ARMS)))
    ax.set_yticklabels(ARMS, fontsize=9)

    # Annotate cell values and box the plan-expected diagonal winners.
    for i, arm in enumerate(ARMS):
        for j, subset in enumerate(SUBSET_ORDER):
            val = matrix.values[i, j]
            # Text color flips at darker reds for legibility.
            txt_color = "white" if val > 0.40 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=txt_color,
            )
            if subset in DIAGONAL[arm]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.0,
                    )
                )

    # Mark which cell wins each column (per-column argmax → green star).
    col_winners = matrix.values.argmax(axis=0)
    for j, i in enumerate(col_winners):
        ax.scatter(
            j, i, marker="*", s=110, color="lime", edgecolor="black", linewidths=0.8
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"AUPRC ({label}) at step-{FINAL_STEP}", fontsize=9)
    ax.set_xlabel("mendelian subset")
    ax.set_ylabel("training arm")
    ax.set_title(
        f"exp187 — Mendelian AUPRC heatmap ({label}, step-{FINAL_STEP})\n"
        f"baseline = {AUPRC_BASELINE:.2f}; black box = plan-expected winner; "
        "green star = actual column winner",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, out_dir, f"exp187_auprc_heatmap_{label.lower()}")


# ---------------------------------------------------------------------------
# Plot 3: training loss curves
# ---------------------------------------------------------------------------


def plot_train_loss(train_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for arm in ARMS:
        arm_hist = (
            train_df[train_df["arm"] == arm][["step", "train/loss"]]
            .dropna()
            .sort_values("step")
        )
        if arm_hist.empty:
            continue
        x = arm_hist["step"].to_numpy()
        y = arm_hist["train/loss"].to_numpy()
        # Legend label includes the # of full passes through the arm's HF
        # dataset at 5K steps × batch 8192 × 256 tokens (≈10.5e9 tokens) —
        # the overfitting risk reads directly from the epoch count.
        legend = f"{arm} ({ARM_EPOCHS[arm]:.2f} epochs)"
        ax.plot(x, y, color=ARM_COLORS[arm], alpha=0.18, linewidth=0.6)
        if len(y) >= 50:
            smooth = pd.Series(y).rolling(50, min_periods=10).mean().to_numpy()
            ax.plot(x, smooth, color=ARM_COLORS[arm], linewidth=2.0, label=legend)
        else:
            ax.plot(x, y, color=ARM_COLORS[arm], linewidth=2.0, label=legend)

    ax.set_xlabel("training step")
    ax.set_ylabel("train/loss (bf16 compute, fp32 params)")
    ax.set_title(
        "exp187 — Training loss per arm (raw faint + 50-step rolling mean bold)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0.4, 2.0)
    ax.set_xlim(0, 5000)
    fig.tight_layout()
    _savefig(fig, out_dir, "exp187_train_loss")


# ---------------------------------------------------------------------------
# Plot 4: LL-gap training curves (in-training, from WandB)
# ---------------------------------------------------------------------------


def plot_llgap_curves(eval_df: pd.DataFrame, out_dir: Path) -> None:
    """One panel per `val_*` recipe; one line per arm × in-training eval step.

    Mirrors the AUPRC-curves layout (subset-as-panel, arm-as-line) but for
    the cheap in-training LL gap signal. Useful for spotting overfitting:
    the LL gap on a recipe that doesn't grow / dips at later steps means
    the arm is no longer learning subset-specific structure.
    """
    n_cols = min(len(VAL_RECIPES), 3)
    n_rows = (len(VAL_RECIPES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    for i, recipe in enumerate(VAL_RECIPES):
        ax = axes[i // n_cols][i % n_cols]
        col = f"{recipe}_llgap"
        for arm in ARMS:
            arm_df = eval_df[
                (eval_df["arm"] == arm) & eval_df[col].notna()
            ].sort_values("step")[["step", col]]
            if arm_df.empty:
                continue
            ax.plot(
                arm_df["step"].to_numpy(),
                arm_df[col].to_numpy(),
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=ARM_COLORS[arm],
                label=arm,
                alpha=0.9,
            )
        ax.axhline(0.0, linestyle=":", color="gray", linewidth=0.8)
        ax.set_xlabel("training step")
        ax.set_ylabel(f"LL gap on `{recipe}` (nat/token)")
        ax.set_title(recipe, fontsize=10)
        ax.grid(True, alpha=0.3)

    for j in range(len(VAL_RECIPES), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    handles, labels_l = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_l,
        loc="lower center",
        ncol=len(ARMS),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )
    fig.suptitle(
        "exp187 — In-training LL gap per `val_*` recipe (WandB)\n"
        "positive = model loses more on shuffled non-functional positions than on functional ones (= good)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _savefig(fig, out_dir, "exp187_llgap_training_curves")


# ---------------------------------------------------------------------------
# Plot 4: LL-gap vs PA scatter with sigmoid fit (issue #8 style)
# ---------------------------------------------------------------------------


def _llgap_at_step(
    eval_df: pd.DataFrame, arm: str, recipe: str, step: int
) -> float | None:
    col = f"{recipe}_llgap"
    if col not in eval_df.columns:
        return None
    sub = eval_df[(eval_df["arm"] == arm) & eval_df[col].notna()][["step", col]]
    if sub.empty:
        return None
    sub = sub.assign(diff=(sub["step"] - step).abs())
    closest = sub.loc[sub["diff"].idxmin()]
    if closest["diff"] > 50:
        return None
    return float(closest[col])


def plot_llgap_vs_auprc(
    pa_df: pl.DataFrame, eval_df: pd.DataFrame, out_dir: Path
) -> None:
    auprc_llr = pa_df.filter(pl.col("score_type") == "minus_llr_avg").to_pandas()

    max_cols = max(len(v) for v in RECIPE_TO_SUBSETS.values())
    n_rows = len(VAL_RECIPES)
    fig, axes = plt.subplots(
        n_rows,
        max_cols,
        figsize=(4 * max_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for i, recipe in enumerate(VAL_RECIPES):
        subsets = RECIPE_TO_SUBSETS[recipe]
        for j in range(max_cols):
            ax = axes[i][j]
            if j >= len(subsets):
                ax.set_visible(False)
                continue
            subset = subsets[j]

            xs, ys, colors, labels_seen = [], [], [], []
            for arm in ARMS:
                for step in STEPS:
                    row = auprc_llr[
                        (auprc_llr["arm"] == arm)
                        & (auprc_llr["step"] == step)
                        & (auprc_llr["subset"] == subset)
                    ]
                    if row.empty:
                        continue
                    auprc = float(row["value"].iloc[0])
                    llgap = _llgap_at_step(eval_df, arm, recipe, step)
                    if llgap is None:
                        continue
                    xs.append(llgap)
                    ys.append(auprc)
                    colors.append(ARM_COLORS[arm])
                    labels_seen.append(arm)

            if not xs:
                ax.set_title(f"{recipe} → {subset}\n(no data)", fontsize=9)
                continue

            xs_a = np.asarray(xs)
            ys_a = np.asarray(ys)

            seen: set[str] = set()
            for x, y, c, lab in zip(xs_a, ys_a, colors, labels_seen):
                show = lab not in seen
                seen.add(lab)
                ax.scatter(
                    x,
                    y,
                    color=c,
                    s=40,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.6,
                    label=lab if show else None,
                )

            popt = _fit_sigmoid(xs_a, ys_a)
            if popt is not None:
                x_fit = np.linspace(xs_a.min(), xs_a.max(), 200)
                ax.plot(
                    x_fit, _sigmoid(x_fit, *popt), color="C3", linewidth=1.5, alpha=0.8
                )
            pearson_r = stats.pearsonr(xs_a, ys_a)
            spearman_r = stats.spearmanr(xs_a, ys_a)
            star_p = (
                "*" if pearson_r.pvalue / 2 < 0.05 and pearson_r.statistic > 0 else ""
            )
            star_s = (
                "*" if spearman_r.pvalue / 2 < 0.05 and spearman_r.statistic > 0 else ""
            )
            ax.text(
                0.04,
                0.96,
                f"r={pearson_r.statistic:.2f}{star_p}\nρ={spearman_r.statistic:.2f}{star_s}\nn={len(xs_a)}",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=2),
            )

            ax.axhline(AUPRC_BASELINE, linestyle=":", color="gray", linewidth=0.8)
            ax.set_xlabel(f"in-training LL gap on `{recipe}` (nat/token)")
            ax.set_ylabel(f"AUPRC (LLR) on `{subset}`")
            ax.set_title(f"{recipe} → {subset}", fontsize=10)
            ax.grid(True, alpha=0.3)

    handles, labels_l = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_l,
        loc="lower center",
        ncol=len(ARMS),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=10,
    )
    fig.suptitle(
        "exp187 — In-training LL gap vs offline AUPRC (red = sigmoid fit; * = one-sided p<0.05)\n"
        f"30 points per panel (6 arms × 5 steps); 1:9 baseline = {AUPRC_BASELINE:.2f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _savefig(fig, out_dir, "exp187_llgap_vs_auprc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading offline AUPRC parquets from S3 (30 files) ...")
    pa_df = load_offline_pa()
    print(f"  rows: {pa_df.height}")
    print(f"  score_types present: {sorted(pa_df['score_type'].unique().to_list())}")

    # AUPRC training curves × {LLR, JSD} × {with-err, no-err}.
    for score_type, label in SCORE_TYPES:
        for with_err in (True, False):
            plot_auprc_curves(
                pa_df, score_type, label, out_dir, with_errorbars=with_err
            )
        plot_auprc_heatmap(pa_df, score_type, label, out_dir)

    print("Fetching wandb training history (6 runs) ...")
    train_df, eval_df = load_wandb_metrics()
    print(f"  train rows: {len(train_df)}, eval rows: {len(eval_df)}")

    plot_train_loss(train_df, out_dir)
    plot_llgap_curves(eval_df, out_dir)
    plot_llgap_vs_auprc(pa_df, eval_df, out_dir)

    print("All plots written to", out_dir)


if __name__ == "__main__":
    main()
