"""issue #302 — iteration 1: missense VEP error forensics across the scaling-v0.5 ladder.

Re-analysis of EXISTING per-variant evals_v2 Mendelian-train scores (no model
re-scoring). Three blocks from the issue-#302 battery:

  0  reproduce & sanity-gate   — per-size AUPRC on minus_llr_avg reproduces the
                                 #279 table; label/score sign; cross-size LLR corr.
  A  error-set forensics       — top false positives / false negatives per size;
                                 how the FP-prone negative set's AF/phyloP profile
                                 shifts with scale; FP-set overlap 128M vs 4B.
  C  pathogenic-vs-benign LLR  — does the pos/neg score separation COMPRESS with
     separation                  scale for missense only (the #296 open question)?
                                 AUROC, Cohen's d, right-tail intrusion.

Focus subset: missense_variant; synonymous_variant + splicing are the within-CDS
contrasts (they IMPROVE with scale). minus_llr_avg = -(llr_fwd + llr_rc)/2 (the
official FWD/RC-averaged -LLR; block 0 confirms it reproduces evals_v2 to 1e-5).

Data (all on S3, train split):
  scores   s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/<model>/mendelian_traits.parquet
  metrics  .../results/metrics/<model>/mendelian_traits.parquet   (official AUPRC, block-0 anchor)
  phyloP   s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits/phyloP_241m_train.parquet

Outputs (scratch/issue302/figs/, PNG 130dpi + SVG):
  auprc_vs_scale          context: AUPRC vs size, 3 subsets (reproduces #279)
  llr_separation_vs_scale C: Cohen's d + frac-neg-above-pos-p90 vs size, 3 subsets
  score_hists             C: pos vs neg minus_llr distributions @ 128M/1B/4B
  fp_scatter_128M_vs_4B   A: missense negatives' 128M vs 4B score, colored by phyloP
Prints: block-0 table, separation table, top-20 missense FP/FN @4B, FP-overlap.

Run:  uv run python scripts/issue302/iter1_ladder_forensics.py
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

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODELS: list[tuple[str, int, str]] = [  # (size label, params M, model dir)
    ("46M", 46, "scaling-v0.5-h640-p46M-step-215573"),
    ("76M", 76, "scaling-v0.5-h768-p76M-step-215573"),
    ("128M", 128, "scaling-v0.5-h896-p128M-step-215573"),
    ("255M", 255, "scaling-v0.5-h1152-p255M-step-215573"),
    ("476M", 476, "scaling-v0.5-h1408-p476M-step-215573"),
    ("1B", 1120, "scaling-v0.5-h1920-p1B-step-215573"),
    ("2B", 2270, "scaling-v0.5-h2432-p2B-step-215573"),
    ("4B", 4020, "scaling-v0.5-h2944-p4B-step-215573"),
]
SIZES = [m[0] for m in MODELS]
PARAMS = {m[0]: m[1] for m in MODELS}
METR = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCOR = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
PHYLOP = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits/phyloP_241m_train.parquet"
SUBSETS = ["missense_variant", "synonymous_variant", "splicing"]
SHORT = {
    "missense_variant": "missense",
    "synonymous_variant": "synonymous",
    "splicing": "splicing",
}
COMMON_AF = 0.01  # "common" negative threshold

OUT = Path("scratch/issue302/figs")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("scratch/issue302/combined_scores.parquet")
KEY = ["chrom", "pos", "ref", "alt"]


# --------------------------------------------------------------------------- #
# Load: combined long-format per-variant scores + phyloP, all sizes
# --------------------------------------------------------------------------- #
def load_combined() -> pl.DataFrame:
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
        "consequence",
        "clinvar_id",
        "exon_closest_pc_gene_id",
        "distance_exon_pc",
    ]
    frames = []
    for lab, _, d in MODELS:
        df = (
            pl.read_parquet(f"{SCOR}/{d}/mendelian_traits.parquet")
            .filter(pl.col("subset").is_in(SUBSETS))
            .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"))
            .select([*keep, "mll"])
            .join(phy, on=KEY, how="left")
            .with_columns(pl.lit(lab).alias("size"))
        )
        frames.append(df)
        print(f"  loaded {lab:>4}  rows={df.height}")
    out = pl.concat(frames)
    n_missing_phy = out.filter(pl.col("phyloP").is_null()).height
    print(f"  phyloP join: {n_missing_phy} / {out.height} rows missing (kept)")
    out.write_parquet(CACHE)
    return out


# --------------------------------------------------------------------------- #
# Block 0 — reproduce & sanity-gate
# --------------------------------------------------------------------------- #
def block0(df: pl.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("BLOCK 0 — reproduce & sanity-gate")
    print("=" * 70)
    # official metrics
    official = {}
    for lab, _, d in MODELS:
        m = pl.read_parquet(f"{METR}/{d}/mendelian_traits.parquet").filter(
            pl.col("score_type") == "minus_llr_avg"
        )
        for s in SUBSETS:
            official[(lab, s)] = m.filter(pl.col("subset") == s)["value"][0]
    print(
        f"\n{'size':>5} | "
        + " | ".join(f"{SHORT[s]:>10}" for s in SUBSETS)
        + "   (recomputed AP; Δ vs official)"
    )
    maxdiff = 0.0
    for lab in SIZES:
        row = []
        for s in SUBSETS:
            sub = df.filter((pl.col("size") == lab) & (pl.col("subset") == s))
            ap = average_precision_score(sub["label"].to_numpy(), sub["mll"].to_numpy())
            maxdiff = max(maxdiff, abs(ap - official[(lab, s)]))
            row.append(f"{ap:10.3f}")
        print(f"{lab:>5} | " + " | ".join(row))
    print(f"\n  max |recomputed AP - official AUPRC| = {maxdiff:.6f}  (gate: < 1e-3)")
    assert maxdiff < 1e-3, "AP does not reproduce official AUPRC"
    # cross-size correlation on missense vs 128M
    print("\n  missense per-variant minus_llr correlation vs 128M:")
    base = df.filter(
        (pl.col("size") == "128M") & (pl.col("subset") == "missense_variant")
    ).select([*KEY, pl.col("mll").alias("m0")])
    for lab in ["255M", "476M", "1B", "2B", "4B"]:
        o = df.filter(
            (pl.col("size") == lab) & (pl.col("subset") == "missense_variant")
        ).select([*KEY, "mll"])
        j = base.join(o, on=KEY, how="inner")
        print(
            f"    128M vs {lab:>4}: Pearson={pearsonr(j['m0'], j['mll'])[0]:.3f} "
            f"Spearman={spearmanr(j['m0'], j['mll'])[0]:.3f}"
        )


# --------------------------------------------------------------------------- #
# Block C — pathogenic-vs-benign LLR separation across scale
# --------------------------------------------------------------------------- #
def separation_stats(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pos, neg = score[label == 1], score[label == 0]
    sd = np.sqrt(
        ((pos.var(ddof=1) * (len(pos) - 1)) + (neg.var(ddof=1) * (len(neg) - 1)))
        / (len(pos) + len(neg) - 2)
    )
    p90 = np.percentile(pos, 90)
    return {
        "auroc": roc_auc_score(label, score),
        "auprc": average_precision_score(label, score),
        "cohen_d": (pos.mean() - neg.mean()) / sd,
        "mean_gap": pos.mean() - neg.mean(),
        "frac_neg_above_pos_p90": float((neg > p90).mean()),
    }


def blockC(df: pl.DataFrame) -> pl.DataFrame:
    print("\n" + "=" * 70)
    print("BLOCK C — pathogenic-vs-benign minus_llr separation vs scale")
    print("=" * 70)
    rows = []
    for s in SUBSETS:
        for lab in SIZES:
            sub = df.filter((pl.col("size") == lab) & (pl.col("subset") == s))
            st = separation_stats(sub["label"].to_numpy(), sub["mll"].to_numpy())
            rows.append({"subset": SHORT[s], "size": lab, "params": PARAMS[lab], **st})
    res = pl.DataFrame(rows)
    for s in ["missense", "synonymous", "splicing"]:
        print(f"\n  {s}:")
        print(
            f"    {'size':>5} {'AUPRC':>7} {'AUROC':>7} {'cohen_d':>8} {'mean_gap':>9} {'neg>posP90':>11}"
        )
        for r in res.filter(pl.col("subset") == s).iter_rows(named=True):
            print(
                f"    {r['size']:>5} {r['auprc']:7.3f} {r['auroc']:7.3f} {r['cohen_d']:8.3f} "
                f"{r['mean_gap']:9.3f} {r['frac_neg_above_pos_p90']:11.3f}"
            )
    return res


# --------------------------------------------------------------------------- #
# Block A — error-set forensics
# --------------------------------------------------------------------------- #
def blockA(df: pl.DataFrame) -> pl.DataFrame:
    print("\n" + "=" * 70)
    print("BLOCK A — error-set forensics (missense)")
    print("=" * 70)
    mis = df.filter(pl.col("subset") == "missense_variant")

    # FP-prone set = top-decile of NEGATIVES by mll, per size; profile its AF/phyloP
    print(
        "\n  FP-prone negatives = top-10% of missense negatives by minus_llr, per size:"
    )
    print(
        f"    {'size':>5} {'n':>4} {'mean_AF':>8} {'frac_common':>11} {'mean_phyloP':>11} {'med_phyloP':>10}"
    )
    prof_rows = []
    for lab in SIZES:
        neg = mis.filter((pl.col("size") == lab) & (pl.col("label") == 0))
        thr = neg["mll"].quantile(0.90)
        top = neg.filter(pl.col("mll") >= thr)
        af = top["AF"].mean()
        frac_common = float((top["AF"] >= COMMON_AF).mean())
        ph = top["phyloP"].drop_nulls()
        prof_rows.append(
            {
                "size": lab,
                "params": PARAMS[lab],
                "n": top.height,
                "mean_AF": af,
                "frac_common": frac_common,
                "mean_phyloP": float(ph.mean()),
                "med_phyloP": float(ph.median()),
            }
        )
        print(
            f"    {lab:>5} {top.height:>4} {af:8.4f} {frac_common:11.3f} "
            f"{float(ph.mean()):11.2f} {float(ph.median()):10.2f}"
        )
    # baseline: all negatives / all positives phyloP+AF (size-invariant set)
    negall = mis.filter((pl.col("size") == "128M") & (pl.col("label") == 0))
    posall = mis.filter((pl.col("size") == "128M") & (pl.col("label") == 1))
    print(
        f"    {'ALL neg':>5} mean_AF={negall['AF'].mean():.4f} mean_phyloP={float(negall['phyloP'].drop_nulls().mean()):.2f} "
        f"med_phyloP={float(negall['phyloP'].drop_nulls().median()):.2f}"
    )
    print(
        f"    {'ALL pos':>5} mean_AF={posall['AF'].mean():.4f} mean_phyloP={float(posall['phyloP'].drop_nulls().mean()):.2f} "
        f"med_phyloP={float(posall['phyloP'].drop_nulls().median()):.2f}"
    )
    prof = pl.DataFrame(prof_rows)

    # FP-set overlap: top-100 negatives by mll, 128M vs 4B
    def topneg(lab, n=100):
        neg = (
            mis.filter((pl.col("size") == lab) & (pl.col("label") == 0))
            .sort("mll", descending=True)
            .head(n)
        )
        return set(zip(neg["chrom"], neg["pos"], neg["ref"], neg["alt"]))

    s128, s4b = topneg("128M"), topneg("4B")
    jac = len(s128 & s4b) / len(s128 | s4b)
    print(
        f"\n  top-100 FP overlap 128M vs 4B: |∩|={len(s128 & s4b)} Jaccard={jac:.3f} "
        f"(low ⇒ 4B makes qualitatively different over-confident calls)"
    )

    # Top-20 missense FPs at 4B, with cross-size scores
    wide = mis.filter(pl.col("label") == 0).pivot(
        values="mll",
        index=[*KEY, "AF", "phyloP", "clinvar_id", "exon_closest_pc_gene_id"],
        on="size",
        aggregate_function="first",
    )
    fp20 = wide.sort("4B", descending=True).head(20)
    print("\n  Top-20 missense FALSE POSITIVES at 4B (benign scored most pathogenic):")
    print(
        f"    {'variant':>20} {'AF':>7} {'phyloP':>6} {'128M':>7} {'1B':>7} {'4B':>7}  gene"
    )
    for r in fp20.iter_rows(named=True):
        v = f"{r['chrom']}:{r['pos']}{r['ref']}>{r['alt']}"
        cv = "" if r["clinvar_id"] is None else " [clinvar]"
        ph = "  n/a" if r["phyloP"] is None else f"{r['phyloP']:6.2f}"
        print(
            f"    {v:>20} {r['AF']:7.4f} {ph} {r['128M']:7.1f} {r['1B']:7.1f} {r['4B']:7.1f}  "
            f"{r['exon_closest_pc_gene_id']}{cv}"
        )

    # Top-20 missense FNs at 4B
    widep = mis.filter(pl.col("label") == 1).pivot(
        values="mll",
        index=[*KEY, "AF", "phyloP", "clinvar_id", "exon_closest_pc_gene_id"],
        on="size",
        aggregate_function="first",
    )
    fn20 = widep.sort("4B", descending=False).head(20)
    print(
        "\n  Top-20 missense FALSE NEGATIVES at 4B (pathogenic scored least pathogenic):"
    )
    print(
        f"    {'variant':>20} {'AF':>9} {'phyloP':>6} {'128M':>7} {'1B':>7} {'4B':>7}  gene"
    )
    for r in fn20.iter_rows(named=True):
        v = f"{r['chrom']}:{r['pos']}{r['ref']}>{r['alt']}"
        cv = "" if r["clinvar_id"] is None else " [clinvar]"
        ph = "  n/a" if r["phyloP"] is None else f"{r['phyloP']:6.2f}"
        print(
            f"    {v:>20} {r['AF']:9.2e} {ph} {r['128M']:7.1f} {r['1B']:7.1f} {r['4B']:7.1f}  "
            f"{r['exon_closest_pc_gene_id']}{cv}"
        )
    return prof


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_auprc(res: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for s, c in [
        ("missense", "tab:blue"),
        ("synonymous", "tab:green"),
        ("splicing", "tab:orange"),
    ]:
        r = res.filter(pl.col("subset") == s).sort("params")
        ax.plot(r["params"], r["auprc"], "o-", color=c, label=s)
    ax.set_xscale("log")
    ax.set_xlabel("params (M, log)")
    ax.set_ylabel("AUPRC (minus_llr_avg)")
    ax.set_title("CDS VEP vs scale — missense degrades, syn/splice improve (#279)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "auprc_vs_scale")


def fig_separation(res: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for s, c in [
        ("missense", "tab:blue"),
        ("synonymous", "tab:green"),
        ("splicing", "tab:orange"),
    ]:
        r = res.filter(pl.col("subset") == s).sort("params")
        axes[0].plot(r["params"], r["cohen_d"], "o-", color=c, label=s)
        axes[1].plot(r["params"], r["frac_neg_above_pos_p90"], "o-", color=c, label=s)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("params (M, log)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Cohen's d  (pos − neg separation)")
    axes[0].set_title("pos/neg separation vs scale")
    axes[1].set_ylabel("frac negatives > positive p90")
    axes[1].set_title("negative right-tail intrusion vs scale")
    fig.suptitle(
        "Block C — does pathogenic-vs-benign minus_llr COMPRESS with scale?", y=1.02
    )
    _save(fig, "llr_separation_vs_scale")


def fig_hists(df: pl.DataFrame) -> None:
    show = ["128M", "1B", "4B"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex="col")
    for j, s in enumerate(SUBSETS):
        for i, lab in enumerate(show):
            ax = axes[i][j]
            sub = df.filter((pl.col("size") == lab) & (pl.col("subset") == s))
            neg = sub.filter(pl.col("label") == 0)["mll"].to_numpy()
            pos = sub.filter(pl.col("label") == 1)["mll"].to_numpy()
            lo, hi = np.percentile(np.concatenate([neg, pos]), [0.5, 99.5])
            bins = np.linspace(lo, hi, 60)
            ax.hist(
                neg,
                bins=bins,
                density=True,
                alpha=0.5,
                color="tab:purple",
                label="benign (neg)",
            )
            ax.hist(
                pos,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.8,
                color="tab:red",
                label="pathogenic (pos)",
            )
            if i == 0:
                ax.set_title(SHORT[s])
            if j == 0:
                ax.set_ylabel(f"{lab}\ndensity")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        "Block C — minus_llr distributions (pos vs neg) across scale; watch the negative right-tail",
        y=1.0,
    )
    _save(fig, "score_hists")


def fig_fp_scatter(df: pl.DataFrame) -> None:
    """Missense negatives: 128M score vs 4B score, colored by phyloP. Points high on
    y but low on x = benign variants 4B newly inflates into the pathogenic range; if
    they are warm (high phyloP) that is the 'conserved-looking benign' failure."""
    mis = df.filter(pl.col("subset") == "missense_variant")
    neg = (
        mis.filter(pl.col("label") == 0)
        .pivot(
            values="mll", index=[*KEY, "phyloP"], on="size", aggregate_function="first"
        )
        .drop_nulls(["128M", "4B"])
    )
    pos4b = mis.filter((pl.col("size") == "4B") & (pl.col("label") == 1))[
        "mll"
    ].to_numpy()
    pos128 = mis.filter((pl.col("size") == "128M") & (pl.col("label") == 1))[
        "mll"
    ].to_numpy()
    p90_4b, p90_128 = np.percentile(pos4b, 90), np.percentile(pos128, 90)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(
        neg["128M"],
        neg["4B"],
        c=neg["phyloP"],
        cmap="viridis",
        s=9,
        alpha=0.6,
        vmin=-2,
        vmax=9,
    )
    lim = [
        min(neg["128M"].min(), neg["4B"].min()) - 2,
        max(neg["128M"].max(), neg["4B"].max()) + 2,
    ]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label="y = x (no change)")
    ax.axhline(
        p90_4b, color="tab:red", lw=1, ls=":", label=f"4B pathogenic p90 = {p90_4b:.0f}"
    )
    ax.axvline(
        p90_128,
        color="tab:blue",
        lw=1,
        ls=":",
        label=f"128M pathogenic p90 = {p90_128:.0f}",
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("minus_llr @ 128M (the missense 'winner')")
    ax.set_ylabel("minus_llr @ 4B")
    ax.set_title(
        "Block A — missense BENIGN negatives: 128M vs 4B score\n"
        "upper-left + warm = conserved benign variants 4B over-inflates"
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("phyloP_241m")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)
    _save(fig, "fp_scatter_128M_vs_4B")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{png,svg}}")


# --------------------------------------------------------------------------- #
def main() -> None:
    print("Loading combined per-variant scores + phyloP ...")
    df = load_combined()
    block0(df)
    resC = blockC(df)
    profA = blockA(df)
    print("\nFigures:")
    fig_auprc(resC)
    fig_separation(resC)
    fig_hists(df)
    fig_fp_scatter(df)
    _ = profA
    resC.write_parquet("scratch/issue302/separation_table.parquet")
    print("\nDone.")


if __name__ == "__main__":
    main()
