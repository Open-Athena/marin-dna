"""Plots for exp232 per-region Qwen3-0.25B sweep (issue #232) + the LL↔AUPRC
correlation EDA (issue #8).

All 6 arms scored offline through the ``evals_v2`` mendelian-traits pipeline
(BOS-faithful ``minus_llr_avg`` = FWD/RC-averaged −LLR). For the correlation we
use the **full per-arm trajectory** — every logged eval step (where both an HF
checkpoint and an in-training LL gap exist) — so the per-subset correlation has
n≈55 instead of the noisy n=6.

Convention (per #8): **LL = −loss**, always **region-matched** — there is no
combined per-region overall LL (only functional/non-functional splits), and we
deliberately do NOT use the region-agnostic ``eval/loss``. The functional-
constraint **LL gap = LL_functional − LL_non-functional = nonfunc_loss −
func_loss** (positive = higher likelihood on functional/constrained positions).

Outputs under ``plots/output/exp232_per_region/`` (PNG 130dpi + SVG):
  exp232_auprc_heatmap.{png,svg}      final-step diagonal (8 subsets × 6 arms)
  exp232_llgap_vs_auprc.{png,svg}     [#8] 8 per-subset panels, AUPRC vs the
                                      subset's matched-region LL gap, n≈55/panel
  exp232_auprc_trajectory.{png,svg}   8 per-subset panels, specialist AUPRC vs step
Prints: diagonal table, per-arm final macro AUPRC, and the per-subset Pearson r
of each region-matched LL metric (functional / non-functional / gap) vs AUPRC.

Run:  uv run python plots/exp232_per_region.py
"""

from __future__ import annotations

import hashlib
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
    "v4_cds",
    "v4_utr3",
    "v4_ncrna_exon",
    "v4_tss_region_and_utr5",
    "v4_ccre_non_promoter",
    "v4_bg",
]
ARM_SHORT: dict[str, str] = {
    "v4_cds": "cds",
    "v4_utr3": "utr3",
    "v4_ncrna_exon": "ncrna",
    "v4_tss_region_and_utr5": "tss",
    "v4_ccre_non_promoter": "ccre",
    "v4_bg": "bg",
}
# Per-arm usable steps = each arm's logged eval/HF-save cadence (every such step
# has both an HF checkpoint and an in-training LL gap). Preemption-offset, so they
# differ per arm (e.g. cds at 1500/2500, not 1000/2000). The full set = the #8
# AUPRC↔LL-gap trajectory; step 4999 alone = the #232 diagonal.
STEPS_BY_ARM: dict[str, list[int]] = {
    "v4_cds": [500, 1500, 2500, 3000, 3500, 4000, 4500, 4999],
    "v4_utr3": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999],
    "v4_ncrna_exon": [500, 1000, 1500, 2000, 3000, 3500, 4000, 4500, 4999],
    "v4_tss_region_and_utr5": [500, 1000, 1500, 2000, 3000, 3500, 4000, 4500, 4999],
    "v4_ccre_non_promoter": [500, 1000, 1500, 2000, 2500, 3000, 4000, 4500, 4999],
    "v4_bg": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999],
}


def arm_steps(arm: str) -> list[int]:
    return STEPS_BY_ARM[arm]


FINAL_STEP: int = 4999
WANDB_PROJECT: str = "marin"
WANDB_GROUP: str = "dna-exp232-v0.1"
S3_PREFIX: str = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"

VAL_RECIPES: list[str] = [
    "val_cds",
    "val_utr3",
    "val_ncrna",
    "val_tss_pc",
    "val_enhancer",
]
RECIPE_TO_SUBSETS: dict[str, list[str]] = {
    "val_cds": ["missense_variant", "synonymous_variant", "splicing"],
    "val_utr3": ["3_prime_UTR_variant"],
    "val_ncrna": ["non_coding_transcript_exon_variant"],
    "val_tss_pc": ["5_prime_UTR_variant", "tss_proximal"],
    "val_enhancer": ["distal"],
}
SUBSET_TO_RECIPE: dict[str, str] = {
    s: r for r, ss in RECIPE_TO_SUBSETS.items() for s in ss
}
DIAGONAL: dict[str, set[str]] = {  # arm → its own region's subsets (heatmap boxes)
    "v4_cds": {"missense_variant", "synonymous_variant", "splicing"},
    "v4_utr3": {"3_prime_UTR_variant"},
    "v4_ncrna_exon": {"non_coding_transcript_exon_variant"},
    "v4_tss_region_and_utr5": {"5_prime_UTR_variant", "tss_proximal"},
    "v4_ccre_non_promoter": {"distal"},
    "v4_bg": set(),
}
SUBSET_TO_ARM: dict[str, str] = {s: a for a, ss in DIAGONAL.items() for s in ss}
SUBSET_ORDER: list[str] = [
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "5_prime_UTR_variant",
    "tss_proximal",
    "distal",
]
ARM_COLORS: dict[str, str] = {
    "v4_cds": "#E69F00",
    "v4_utr3": "#56B4E9",
    "v4_ncrna_exon": "#009E73",
    "v4_tss_region_and_utr5": "#F0E442",
    "v4_ccre_non_promoter": "#0072B2",
    "v4_bg": "#D55E00",
}
SCORE_TYPE: str = "minus_llr_avg"
AUPRC_BASELINE: float = 0.10
OUT_DIR: Path = Path(__file__).parent / "output" / Path(__file__).stem
CACHE_DIR: Path = OUT_DIR / ".wandb_cache"


def _savefig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=130)), ("svg", {})):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  wrote {stem}.png + {stem}.svg")


def _sigmoid(x, lo, hi, k, x0):
    """Logistic fit (issue #8 convention): AUPRC saturates in the LL gap."""
    return lo + (hi - lo) / (1 + np.exp(np.clip(-k * (x - x0), -500, 500)))


def _fit_sigmoid(x: np.ndarray, y: np.ndarray):
    """4-param logistic fit; None if it fails to converge."""
    try:
        k0 = 5.0 if np.corrcoef(x, y)[0, 1] >= 0 else -5.0
        p0 = [float(y.min()), float(y.max()), k0, float(np.median(x))]
        bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
        popt, _ = optimize.curve_fit(_sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except (RuntimeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_offline_auprc() -> pl.DataFrame:
    """All (arm, step) AUPRC for SCORE_TYPE. Columns: arm, step, subset, auprc, n_groups."""
    frames, missing = [], []
    for arm in ARMS:
        for step in arm_steps(arm):
            uri = f"{S3_PREFIX}/exp232-{arm}-step-{step}/mendelian_traits.parquet"
            try:
                df = pl.read_parquet(uri).filter(pl.col("score_type") == SCORE_TYPE)
            except (OSError, FileNotFoundError):
                missing.append(f"{ARM_SHORT[arm]}-{step}")
                continue
            frames.append(
                df.select(
                    pl.lit(arm).alias("arm"),
                    pl.lit(step).alias("step"),
                    "subset",
                    pl.col("value").alias("auprc"),
                    "n_groups",
                )
            )
    if missing:
        print(f"  WARNING: {len(missing)} AUPRC parquet(s) missing: {missing}")
    return pl.concat(frames, how="vertical")


def _step_col(hist: pd.DataFrame) -> pd.Series:
    if "global_step" in hist.columns and hist["global_step"].notna().any():
        return hist["global_step"].fillna(hist["_step"]).astype(int)
    return hist["_step"].astype(int)


def load_wandb_trajectory() -> pd.DataFrame:
    """Per (arm, step, recipe): func_ll, nonfunc_ll, gap (LL=-loss). From wandb history."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys = [
        f"eval/{r}_{k}/loss"
        for r in VAL_RECIPES
        for k in ("functional", "nonfunctional")
    ]
    # Key the cache on the inputs that determine its content, so editing ARMS /
    # VAL_RECIPES / WANDB_GROUP invalidates it instead of silently serving stale rows.
    cfg = hashlib.md5(
        repr((ARMS, VAL_RECIPES, WANDB_GROUP, keys)).encode()
    ).hexdigest()[:8]
    cache = CACHE_DIR / f"exp232_llgap_trajectory_{cfg}.parquet"
    if cache.exists():
        print(f"  using cached wandb trajectory ({cache.name})")
        return pd.read_parquet(cache)
    api = wandb.Api()
    runs = list(
        api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP}, order="-created_at")
    )
    out = []
    for arm in ARMS:
        cands = [r for r in runs if f"-{arm}-v0.1" in r.name]
        cands.sort(key=lambda r: (r.state == "finished", r.created_at), reverse=True)
        run = cands[0]
        print(f"  {ARM_SHORT[arm]:6s} history <- {run.name}")
        h = run.history(keys=keys, samples=10000, pandas=True)
        h["arm"] = arm
        h["step"] = _step_col(h)
        for recipe in VAL_RECIPES:
            f, n = f"eval/{recipe}_functional/loss", f"eval/{recipe}_nonfunctional/loss"
            for _, row in h[["arm", "step", f, n]].dropna().iterrows():
                out.append(
                    {
                        "arm": arm,
                        "step": int(row["step"]),
                        "recipe": recipe,
                        "func_ll": -row[f],
                        "nonfunc_ll": -row[n],
                        "gap": row[n] - row[f],
                    }
                )
    df = pd.DataFrame(out)
    df.to_parquet(cache)
    print(f"  cached {len(df)} (arm,step,recipe) rows")
    return df


def _ll_at_step(
    traj: pd.DataFrame, arm: str, recipe: str, metric: str, step: int
) -> float | None:
    sub = traj[(traj["arm"] == arm) & (traj["recipe"] == recipe)]
    if sub.empty:
        return None
    sub = sub.assign(d=(sub["step"] - step).abs())
    row = sub.loc[sub["d"].idxmin()]
    return float(row[metric]) if row["d"] <= 100 else None


# ---------------------------------------------------------------------------
# Plot 1: final-step diagonal heatmap
# ---------------------------------------------------------------------------


def plot_auprc_heatmap(au: pl.DataFrame) -> None:
    fin = au.filter(pl.col("step") == FINAL_STEP)
    lut = {(r["arm"], r["subset"]): r["auprc"] for r in fin.iter_rows(named=True)}
    mat = np.array([[lut.get((a, s), np.nan) for a in ARMS] for s in SUBSET_ORDER])

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(mat, cmap="Reds", aspect="auto", vmin=AUPRC_BASELINE, vmax=0.40)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_SHORT[a] for a in ARMS], fontsize=10)
    ax.set_yticks(range(len(SUBSET_ORDER)))
    ax.set_yticklabels([s.replace("_variant", "") for s in SUBSET_ORDER], fontsize=9)
    for i, sub in enumerate(SUBSET_ORDER):
        for j, arm in enumerate(ARMS):
            v = mat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > 0.28 else "black",
            )
            if sub in DIAGONAL[arm]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.0,
                    )
                )
    for i in range(len(SUBSET_ORDER)):
        j = int(np.nanargmax(mat[i, :]))
        ax.scatter(
            j + 0.30,
            i - 0.30,
            marker="*",
            s=95,
            color="lime",
            edgecolor="black",
            linewidths=0.8,
            zorder=5,
        )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"offline AUPRC ({SCORE_TYPE}), step-{FINAL_STEP}", fontsize=9)
    ax.set_xlabel("training arm (region specialist)")
    ax.set_ylabel("mendelian variant subset")
    ax.set_title(
        "exp232 — per-region diagonal: offline Mendelian AUPRC\n"
        f"baseline={AUPRC_BASELINE:.2f}; black box = arm's own region; green ★ = row winner",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, "exp232_auprc_heatmap")


# ---------------------------------------------------------------------------
# Plot 2: per-subset AUPRC vs matched-region LL gap (trajectory, n≈55) — #8
# ---------------------------------------------------------------------------


def plot_llgap_vs_auprc(au: pl.DataFrame, traj: pd.DataFrame) -> None:
    aulut = {
        (r["arm"], r["step"], r["subset"]): r["auprc"] for r in au.iter_rows(named=True)
    }
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
    )
    per_subset = []
    for idx, sub in enumerate(SUBSET_ORDER):
        ax = axes[idx // ncols][idx % ncols]
        recipe = SUBSET_TO_RECIPE[sub]
        xs, ys = [], []
        for arm in ARMS:
            for step in arm_steps(arm):
                a = aulut.get((arm, step, sub))
                g = _ll_at_step(traj, arm, recipe, "gap", step)
                if a is None or g is None:
                    continue
                ax.scatter(
                    g,
                    a,
                    color=ARM_COLORS[arm],
                    s=42,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                    label=ARM_SHORT[arm],
                    zorder=3,
                )
                xs.append(g)
                ys.append(a)
        xs_a, ys_a = np.array(xs), np.array(ys)
        if len(xs_a) >= 3:
            xf = np.linspace(xs_a.min(), xs_a.max(), 100)
            popt = _fit_sigmoid(xs_a, ys_a)
            if popt is not None:
                ax.plot(xf, _sigmoid(xf, *popt), color="C3", linewidth=1.4, alpha=0.85)
            pr, sr = stats.pearsonr(xs_a, ys_a), stats.spearmanr(xs_a, ys_a)
            per_subset.append((sub, pr.statistic, sr.statistic, len(xs_a)))
            ax.text(
                0.04,
                0.96,
                f"r={pr.statistic:+.2f}\nρ={sr.statistic:+.2f}\nn={len(xs_a)}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=2),
            )
        ax.axhline(AUPRC_BASELINE, linestyle=":", color="gray", linewidth=0.8)
        ax.set_title(f"{sub.replace('_variant', '')}  ←  {recipe}", fontsize=9)
        ax.set_xlabel("matched-region LL gap", fontsize=8)
        ax.set_ylabel("AUPRC", fontsize=8)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    by = dict(zip(labels, handles))
    fig.legend(
        [by[ARM_SHORT[a]] for a in ARMS if ARM_SHORT[a] in by],
        [ARM_SHORT[a] for a in ARMS if ARM_SHORT[a] in by],
        loc="lower center",
        ncol=len(ARMS),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "exp232 / #8 — per-subset AUPRC vs its matched-region functional-constraint LL gap\n"
        "trajectory: 6 arms × all logged eval steps (n≈55/panel); LL = −loss",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _savefig(fig, "exp232_llgap_vs_auprc")
    print("  per-subset (vs matched-region LL gap) Pearson r / Spearman ρ:")
    for sub, pr, sr, n in per_subset:
        print(f"    {sub[:34]:34s} r={pr:+.2f}  ρ={sr:+.2f}  n={n}")


# ---------------------------------------------------------------------------
# Plot 3: specialist AUPRC trajectory per subset (matched arm only)
# ---------------------------------------------------------------------------


def plot_auprc_trajectory(au: pl.DataFrame) -> None:
    """8 panels: AUPRC vs training step for the single specialist arm matched to
    each subset (its diagonal arm). Shows convergence / overfitting per region."""
    lut = {
        (r["arm"], r["step"], r["subset"]): r["auprc"] for r in au.iter_rows(named=True)
    }
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False
    )
    for idx, sub in enumerate(SUBSET_ORDER):
        ax = axes[idx // ncols][idx % ncols]
        arm = SUBSET_TO_ARM[sub]
        xs = [s for s in arm_steps(arm) if (arm, s, sub) in lut]
        ys = [lut[(arm, s, sub)] for s in xs]
        ax.plot(xs, ys, marker="o", color=ARM_COLORS[arm], linewidth=2, markersize=5)
        ax.axhline(AUPRC_BASELINE, linestyle=":", color="gray", linewidth=0.8)
        ax.set_title(f"{sub.replace('_variant', '')}  ({ARM_SHORT[arm]})", fontsize=9)
        ax.set_xlabel("training step", fontsize=8)
        ax.set_ylabel("AUPRC", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "exp232 — specialist AUPRC trajectory per subset (the matched region arm only)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, "exp232_auprc_trajectory")


# ---------------------------------------------------------------------------
# Tables: diagonal + region-matched LL-metric comparison
# ---------------------------------------------------------------------------


def print_tables(au: pl.DataFrame, traj: pd.DataFrame) -> None:
    fin = au.filter(pl.col("step") == FINAL_STEP)
    lut = {(r["arm"], r["subset"]): r["auprc"] for r in fin.iter_rows(named=True)}
    nlut = {
        r["subset"]: r["n_groups"]
        for r in fin.filter(pl.col("arm") == "v4_cds").iter_rows(named=True)
    }

    print("\n=== DIAGONAL (final step-4999): subset × arm AUPRC, row-winner in [] ===")
    print("subset".ljust(32) + "n   " + " ".join(ARM_SHORT[a].center(7) for a in ARMS))
    for sub in SUBSET_ORDER:
        vals = {a: lut.get((a, sub), float("nan")) for a in ARMS}
        best = max(vals, key=lambda a: vals[a] if vals[a] == vals[a] else -np.inf)
        cells = " ".join(
            (f"[{vals[a]:.3f}]" if a == best else f" {vals[a]:.3f} ").center(7)
            for a in ARMS
        )
        print(f"{sub[:31].ljust(32)}{int(nlut.get(sub, 0)):<4}{cells}")

    macro = {
        r["arm"]: r["auprc"]
        for r in fin.filter(pl.col("subset") == "_macro_avg_").iter_rows(named=True)
    }
    print("\n=== per-arm final macro AUPRC ===")
    for a in sorted(ARMS, key=lambda a: macro.get(a, 0), reverse=True):
        print(f"  {ARM_SHORT[a]:6s} {macro.get(a, float('nan')):.3f}")

    print(
        "\n=== region-matched LL metric vs AUPRC — per-subset Pearson r (trajectory) ==="
    )
    print(f"{'subset':34s}{'LL_func':>9}{'LL_nonf':>9}{'LL_gap':>9}{'n':>5}")
    aulut = {
        (r["arm"], r["step"], r["subset"]): r["auprc"] for r in au.iter_rows(named=True)
    }
    acc = {m: [] for m in ("func_ll", "nonfunc_ll", "gap")}
    for sub in SUBSET_ORDER:
        rec = SUBSET_TO_RECIPE[sub]
        ys, xs = [], {m: [] for m in ("func_ll", "nonfunc_ll", "gap")}
        for arm in ARMS:
            for step in arm_steps(arm):
                a = aulut.get((arm, step, sub))
                if a is None:
                    continue
                vals = {m: _ll_at_step(traj, arm, rec, m, step) for m in xs}
                if any(v is None for v in vals.values()):
                    continue
                ys.append(a)
                for m in xs:
                    xs[m].append(vals[m])
        rs = {m: stats.pearsonr(np.array(xs[m]), np.array(ys)).statistic for m in xs}
        for m in acc:
            acc[m].append(rs[m])
        print(
            f"{sub[:33]:34s}{rs['func_ll']:+9.2f}{rs['nonfunc_ll']:+9.2f}{rs['gap']:+9.2f}{len(ys):>5}"
        )
    print(
        f"{'MEAN':34s}"
        + "".join(f"{np.mean(acc[m]):+9.2f}" for m in ("func_ll", "nonfunc_ll", "gap"))
    )


def main() -> None:
    print("Loading offline AUPRC trajectory (S3) ...")
    au = load_offline_auprc()
    print(f"  {au.height} rows over {sorted(au['step'].unique().to_list())}")
    print("Loading wandb LL-gap trajectory ...")
    traj = load_wandb_trajectory()
    print("Plotting ...")
    plot_auprc_heatmap(au)
    plot_llgap_vs_auprc(au, traj)
    plot_auprc_trajectory(au)
    print_tables(au, traj)
    print(f"\nDone. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
