"""issue #302 — iteration 2: missense error forensics ALONG the #232 0.25B CDS
training trajectory, and a cross-axis test against the iteration-1 ladder.

The #232 `v4_cds` 0.25B specialist is the only arm whose missense AUPRC turns
*down* late in training (peaks ~step 3500-4000, eases to 0.309 @ step 4999). This
is the within-MODEL analogue of the across-SCALE missense degradation that
iteration 1 characterized on the `scaling-v0.5` ladder. Question: does the
late-training decline carry the SAME signature (pos/neg compression + a churning
set of conserved-looking benign FPs), and are the variants the tiny model breaks
on late the SAME ones the 4B breaks on with scale? If yes → one phenomenon
(training-dynamics / over-fitting), not a pure capacity effect.

Points = the 10 `exp232-v4_cds-step-*` checkpoints (per-variant evals_v2
Mendelian-train scores, same schema/variants as the ladder). minus_llr_avg =
-(llr_fwd + llr_rc)/2. Focus: missense; synonymous + splicing the CDS controls.

Cross-axis input: the ladder 128M/4B missense scores cached by iteration 1
(scratch/issue302/combined_scores.parquet — run iter1 first).

Outputs (scratch/issue302/figs/, PNG 130dpi + SVG):
  traj_auprc_vs_step        per-step AUPRC, 3 subsets (reproduces #232 trajectory)
  traj_separation_vs_step   Cohen's d + frac-neg-above-pos-p90 vs step, 3 subsets
  crossaxis_inflation       missense negatives: scale-inflation (4B-128M) vs
                            training-inflation (final-peak), colored by phyloP
Prints: per-step AUPRC + peak detection; separation table; FP-set churn
(peak vs final, and CDS-final vs ladder-4B); cross-axis inflation correlation.

Run:  uv run python scripts/issue302/iter2_cds_trajectory.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999]
SCOR = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
PHYLOP = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits/phyloP_241m_train.parquet"
SUBSETS = ["missense_variant", "synonymous_variant", "splicing"]
SHORT = {
    "missense_variant": "missense",
    "synonymous_variant": "synonymous",
    "splicing": "splicing",
}
KEY = ["chrom", "pos", "ref", "alt"]
COMMON_AF = 0.01

OUT = Path("scratch/issue302/figs")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("scratch/issue302/cds_traj_scores.parquet")
LADDER_CACHE = Path("scratch/issue302/combined_scores.parquet")  # from iter1


def load_traj() -> pl.DataFrame:
    if CACHE.exists():
        print(f"  using cached {CACHE}")
        return pl.read_parquet(CACHE)
    phy = pl.read_parquet(PHYLOP).select([*KEY, pl.col("score").alias("phyloP")])
    keep = [
        *KEY,
        "label",
        "subset",
        "match_group",
        "AF",
        "clinvar_id",
        "exon_closest_pc_gene_id",
    ]
    frames = []
    for st in STEPS:
        df = (
            pl.read_parquet(f"{SCOR}/exp232-v4_cds-step-{st}/mendelian_traits.parquet")
            .filter(pl.col("subset").is_in(SUBSETS))
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"))
            .select([*keep, "mll"])
            .join(phy, on=KEY, how="left")
            .with_columns(pl.lit(st).alias("step"))
        )
        frames.append(df)
        print(f"  loaded step-{st:<4} rows={df.height}")
    out = pl.concat(frames)
    out.write_parquet(CACHE)
    return out


def separation_stats(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pos, neg = score[label == 1], score[label == 0]
    sd = np.sqrt(
        ((pos.var(ddof=1) * (len(pos) - 1)) + (neg.var(ddof=1) * (len(neg) - 1)))
        / (len(pos) + len(neg) - 2)
    )
    p90 = np.percentile(pos, 90)
    return {
        "auprc": average_precision_score(label, score),
        "auroc": roc_auc_score(label, score),
        "cohen_d": (pos.mean() - neg.mean()) / sd,
        "frac_neg_above_pos_p90": float((neg > p90).mean()),
    }


def block0_C(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    print("\n" + "=" * 70)
    print("BLOCK 0/C — per-step AUPRC + pos/neg separation (CDS-arm trajectory)")
    print("=" * 70)
    rows = []
    for s in SUBSETS:
        for st in STEPS:
            sub = df.filter((pl.col("step") == st) & (pl.col("subset") == s))
            rows.append(
                {
                    "subset": SHORT[s],
                    "step": st,
                    **separation_stats(sub["label"].to_numpy(), sub["mll"].to_numpy()),
                }
            )
    res = pl.DataFrame(rows)
    mis = res.filter(pl.col("subset") == "missense")
    peak = int(mis.sort("auprc", descending=True)["step"][0])
    print(f"\n  missense AUPRC trajectory (peak @ step-{peak}):")
    print(
        f"    {'step':>5} {'AUPRC':>7} {'AUROC':>7} {'cohen_d':>8} {'neg>posP90':>11}"
    )
    for r in mis.sort("step").iter_rows(named=True):
        mark = "  <- peak" if r["step"] == peak else ""
        print(
            f"    {r['step']:>5} {r['auprc']:7.3f} {r['auroc']:7.3f} {r['cohen_d']:8.3f} {r['frac_neg_above_pos_p90']:11.3f}{mark}"
        )
    for s in ["synonymous", "splicing"]:
        sr = res.filter(pl.col("subset") == s).sort("step")
        print(
            f"\n  {s}: AUPRC {sr['auprc'][0]:.3f} -> {sr['auprc'][-1]:.3f}  |  cohen_d {sr['cohen_d'][0]:.3f} -> {sr['cohen_d'][-1]:.3f}"
        )
    return res, peak


def blockA(df: pl.DataFrame, peak: int) -> None:
    print("\n" + "=" * 70)
    print("BLOCK A — FP forensics along the trajectory (missense)")
    print("=" * 70)
    mis = df.filter(pl.col("subset") == "missense_variant")
    print("\n  FP-prone negatives (top-10% by score) per step — phyloP/AF profile:")
    print(
        f"    {'step':>5} {'mean_phyloP':>11} {'med_phyloP':>10} {'mean_AF':>8} {'frac_common':>11}"
    )
    for st in STEPS:
        neg = mis.filter((pl.col("step") == st) & (pl.col("label") == 0))
        top = neg.filter(pl.col("mll") >= neg["mll"].quantile(0.90))
        ph = top["phyloP"].drop_nulls()
        print(
            f"    {st:>5} {float(ph.mean()):11.2f} {float(ph.median()):10.2f} "
            f"{top['AF'].mean():8.4f} {float((top['AF'] >= COMMON_AF).mean()):11.3f}"
        )
    negall = mis.filter((pl.col("step") == peak) & (pl.col("label") == 0))
    print(
        f"    ALL neg: mean_phyloP={float(negall['phyloP'].drop_nulls().mean()):.2f} mean_AF={negall['AF'].mean():.4f}"
    )

    def topneg(step, n=100):
        neg = (
            mis.filter((pl.col("step") == step) & (pl.col("label") == 0))
            .sort("mll", descending=True)
            .head(n)
        )
        return set(zip(neg["chrom"], neg["pos"], neg["ref"], neg["alt"]))

    sp, sf = topneg(peak), topneg(4999)
    jac = len(sp & sf) / len(sp | sf)
    print(
        f"\n  top-100 FP churn peak(step-{peak}) vs final(step-4999): |∩|={len(sp & sf)} Jaccard={jac:.3f}"
    )


def cross_axis(df: pl.DataFrame, peak: int) -> pl.DataFrame:
    """Are the variants the CDS arm over-inflates LATE the same ones the ladder
    over-inflates with SCALE?  Correlate per-negative training-inflation
    (mll@4999 - mll@peak) with scale-inflation (mll@4B - mll@128M)."""
    print("\n" + "=" * 70)
    print("CROSS-AXIS — within-training (0.25B) vs across-scale (ladder) inflation")
    print("=" * 70)
    if not LADDER_CACHE.exists():
        print("  ladder cache missing — run iter1 first; skipping cross-axis.")
        return pl.DataFrame()
    lad = (
        pl.read_parquet(LADDER_CACHE)
        .filter(
            (pl.col("subset") == "missense_variant")
            & (pl.col("label") == 0)
            & (pl.col("size").is_in(["128M", "4B"]))
        )
        .select([*KEY, "size", "mll", "phyloP", "AF"])
        .pivot(
            values="mll",
            index=[*KEY, "phyloP", "AF"],
            on="size",
            aggregate_function="first",
        )
        .with_columns((pl.col("4B") - pl.col("128M")).alias("scale_infl"))
    )
    cds = (
        df.filter(
            (pl.col("subset") == "missense_variant")
            & (pl.col("label") == 0)
            & (pl.col("step").is_in([peak, 4999]))
        )
        .select([*KEY, "step", "mll"])
        .pivot(values="mll", index=KEY, on="step", aggregate_function="first")
        .rename({str(peak): "cds_peak", "4999": "cds_final"})
        .with_columns((pl.col("cds_final") - pl.col("cds_peak")).alias("train_infl"))
    )
    j = lad.join(cds, on=KEY, how="inner").drop_nulls(["scale_infl", "train_infl"])
    pr = pearsonr(j["scale_infl"], j["train_infl"])
    sr = spearmanr(j["scale_infl"], j["train_infl"])
    print(f"\n  missense negatives n={j.height}")
    print(
        f"  corr(scale_infl 4B-128M, train_infl 4999-peak): Pearson={pr[0]:+.3f} (p={pr[1]:.1e})  Spearman={sr[0]:+.3f}"
    )
    # FP-set overlap: CDS-final top-100 vs ladder-4B top-100 (by respective scores)
    cds_fp = set(
        zip(*j.sort("cds_final", descending=True).head(100).select(KEY).to_numpy().T)
    )
    lad_fp = set(zip(*j.sort("4B", descending=True).head(100).select(KEY).to_numpy().T))
    jac = len(cds_fp & lad_fp) / len(cds_fp | lad_fp)
    print(
        f"  top-100 FP overlap CDS-arm@4999 vs ladder@4B: |∩|={len(cds_fp & lad_fp)} Jaccard={jac:.3f}"
    )
    # both-inflated set: high on both axes — profile
    hi = j.filter(
        (pl.col("scale_infl") > j["scale_infl"].quantile(0.9))
        & (pl.col("train_infl") > j["train_infl"].quantile(0.9))
    )
    print(
        f"  negatives in the top-decile of BOTH inflations: n={hi.height} "
        f"mean_phyloP={float(hi['phyloP'].drop_nulls().mean()):.2f} mean_AF={hi['AF'].mean():.4f} "
        f"frac_common={float((hi['AF'] >= COMMON_AF).mean()):.3f}"
    )
    return j


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_traj_auprc(res: pl.DataFrame, peak: int) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for s, c in [
        ("missense", "tab:blue"),
        ("synonymous", "tab:green"),
        ("splicing", "tab:orange"),
    ]:
        r = res.filter(pl.col("subset") == s).sort("step")
        ax.plot(r["step"], r["auprc"], "o-", color=c, label=s)
    ax.axvline(
        peak,
        color="tab:blue",
        ls=":",
        lw=1,
        alpha=0.7,
        label=f"missense peak (step {peak})",
    )
    ax.set_xlabel("training step (0.25B CDS arm)")
    ax.set_ylabel("AUPRC (minus_llr_avg)")
    ax.set_title("#232 0.25B CDS arm — missense AUPRC turns down late")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "traj_auprc_vs_step")


def fig_traj_sep(res: pl.DataFrame, peak: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4))
    for s, c in [
        ("missense", "tab:blue"),
        ("synonymous", "tab:green"),
        ("splicing", "tab:orange"),
    ]:
        r = res.filter(pl.col("subset") == s).sort("step")
        axes[0].plot(r["step"], r["cohen_d"], "o-", color=c, label=s)
        axes[1].plot(r["step"], r["frac_neg_above_pos_p90"], "o-", color=c, label=s)
    for ax in axes:
        ax.axvline(peak, color="tab:blue", ls=":", lw=1, alpha=0.6)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cohen's d (pos − neg)")
    axes[0].set_title("pos/neg separation vs training step")
    axes[1].set_ylabel("frac negatives > positive p90")
    axes[1].set_title("negative right-tail intrusion vs training step")
    fig.suptitle(
        "Block C (within-training) — does missense separation compress LATE?", y=1.02
    )
    _save(fig, "traj_separation_vs_step")


def fig_crossaxis(j: pl.DataFrame) -> None:
    if j.is_empty():
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(
        j["scale_infl"],
        j["train_infl"],
        c=j["phyloP"],
        cmap="viridis",
        s=9,
        alpha=0.55,
        vmin=-2,
        vmax=9,
    )
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    pr = pearsonr(j["scale_infl"], j["train_infl"])[0]
    ax.set_xlabel("scale inflation  (minus_llr  4B − 128M)")
    ax.set_ylabel("training inflation  (minus_llr  step4999 − peak)")
    ax.set_title(
        f"Cross-axis: same missense benigns inflated by SCALE and by\nlate TRAINING?  Pearson r = {pr:+.2f}"
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("phyloP_241m")
    ax.grid(alpha=0.25)
    _save(fig, "crossaxis_inflation")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{png,svg}}")


def main() -> None:
    print("Loading CDS-arm trajectory scores + phyloP ...")
    df = load_traj()
    res, peak = block0_C(df)
    blockA(df, peak)
    j = cross_axis(df, peak)
    print("\nFigures:")
    fig_traj_auprc(res, peak)
    fig_traj_sep(res, peak)
    fig_crossaxis(j)
    print("\nDone.")


if __name__ == "__main__":
    main()
