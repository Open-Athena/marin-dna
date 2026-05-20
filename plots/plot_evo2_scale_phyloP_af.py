"""Second iteration for issue #203 — AF-threshold filter + phyloP correlation analysis.

Tests two hypotheses raised on the issue thread:

1. TraitGym-v1-style filter: the previous (pre-PR-#194) dataset had
   negatives at AF ≳ 5%; the new k=9 dataset goes down to AF ≳ 0.1%.
   Filtering the new dataset to AF ≥ {0.5, 1, 5}% should recover
   v1-like AUPRC if low-AF negatives are the driver of the scale-hurt.
2. Phylogenetic-noise memorization: bigger Evo 2 may have internalized
   cross-species conservation more strongly. Measure
   Spearman(phyloP_241m, minus_llr) per model on each subset; if the
   correlation strengthens monotonically with scale, that's direct
   evidence.

Outputs all under `plots/output/evo2_scale_comparison/`:
- `af_threshold_metrics_missense.parquet` — AUPRC + MRR per (af_thr, model)
- `phyloP_score_spearman.parquet` — Spearman per (subset, label, model)
- `af_threshold_missense.{svg,png}` — 3-panel scale curves vs AF threshold
- `phyloP_vs_score_missense.{svg,png}` — 3-panel scatter, model facets

Usage:
    uv run python plots/plot_evo2_scale_phyloP_af.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr

from bolinas.pipelines.evals.metrics import (
    auprc_with_bootstrap_se,
    mrr_within_group,
)

MODELS: tuple[str, ...] = ("evo2_1b_base", "evo2_7b", "evo2_40b")
MODEL_COLORS: dict[str, str] = {
    "evo2_1b_base": "#3b1a64",
    "evo2_7b": "#3a6fa0",
    "evo2_40b": "#e8d840",
}
GIST_URLS: dict[str, str] = {
    "evo2_1b_base": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/b6c254849a71ca0783a24218a4fe9037e887e8f7/evo2_1b_base_train.parquet",
    "evo2_7b": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/e72d3d2e14955a670b8229dc8d525a69ea88c05c/evo2_7b_train.parquet",
    "evo2_40b": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/2b425e759811c201ca806ae4c8733fd7732220a6/evo2_40b_train.parquet",
}
DATASET_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
PHYLOP_S3 = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits/phyloP_241m_train.parquet"
OUT_DIR = Path(__file__).parent / "output" / "evo2_scale_comparison"
AF_THRESHOLDS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)
SUBSETS_FOR_RHO: tuple[str, ...] = (
    "missense_variant",
    "splicing",
    "tss_proximal",
    "non_coding_transcript_exon_variant",
)


def load_annotated() -> pd.DataFrame:
    """Load predictions + AF + phyloP_241m, all keyed on (chrom,pos,ref,alt)."""
    ds = load_dataset(
        "bolinas-dna/evals_mendelian_traits",
        revision=DATASET_REVISION,
        split="train",
    )
    ann = ds.to_pandas()[
        [
            "chrom",
            "pos",
            "ref",
            "alt",
            "AF",
            "consequence_final",
            "exon_closest_pc_gene_id",
        ]
    ]
    ann["chrom"] = ann["chrom"].astype(str)
    phyloP = pd.read_parquet(PHYLOP_S3)[["chrom", "pos", "ref", "alt", "score"]].rename(
        columns={"score": "phyloP"}
    )
    phyloP["chrom"] = phyloP["chrom"].astype(str)

    parts: list[pd.DataFrame] = []
    for model, url in GIST_URLS.items():
        d = pd.read_parquet(url)
        d["chrom"] = d["chrom"].astype(str)
        d = d.merge(ann, on=["chrom", "pos", "ref", "alt"], how="left")
        d = d.merge(phyloP, on=["chrom", "pos", "ref", "alt"], how="left")
        parts.append(d.assign(model=model))
    return pd.concat(parts, ignore_index=True)


def af_threshold_sweep(
    long_df: pd.DataFrame, subset: str = "missense_variant"
) -> pd.DataFrame:
    """For each AF threshold, drop negatives below it (positives kept regardless),
    then drop match_groups that no longer have ≥1 negative. Returns one row per
    (af_thr, model) with AUPRC + MRR (cluster-bootstrap SE)."""
    rows: list[dict] = []
    for thr in (None, *AF_THRESHOLDS):
        sub_all = long_df[long_df["subset"] == subset].copy()
        if thr is not None:
            sub_all = sub_all[(sub_all["label"] == 1) | (sub_all["AF"] >= thr)]
        counts = sub_all.groupby(["match_group", "model"])["label"].agg(
            ["sum", "count"]
        )
        bad = (
            counts[(counts["sum"] != 1) | (counts["count"] < 2)]
            .reset_index()["match_group"]
            .unique()
        )
        sub_all = sub_all[~sub_all["match_group"].isin(bad)]
        n_groups = sub_all[sub_all["model"] == MODELS[0]]["match_group"].nunique()
        if n_groups < 10:
            continue
        n_pos = int((sub_all[sub_all["model"] == MODELS[0]]["label"] == 1).sum())
        n_all = int((sub_all["model"] == MODELS[0]).sum())
        prevalence = n_pos / n_all
        mean_k = (
            sub_all[sub_all["model"] == MODELS[0]].groupby("match_group").size().mean()
            - 1
        )
        for model in MODELS:
            s = sub_all[sub_all["model"] == model]
            auprc = auprc_with_bootstrap_se(
                s["label"], s["minus_llr"], s["match_group"], n_bootstrap=300, rng=0
            )
            mrr = mrr_within_group(
                s["label"], s["minus_llr"], s["match_group"], n_bootstrap=300, rng=0
            )
            rows.append(
                {
                    "af_thr": thr,
                    "model": model,
                    "n_groups": int(n_groups),
                    "mean_k": float(mean_k),
                    "prevalence": float(prevalence),
                    "auprc": auprc["value"],
                    "auprc_se": auprc["se"],
                    "auprc_minus_prev": auprc["value"] - prevalence,
                    "mrr": mrr["value"],
                    "mrr_se": mrr["se"],
                }
            )
    return pd.DataFrame(rows)


def phyloP_spearman(long_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman(phyloP_241m, minus_llr) per (subset, label, model)."""
    rows: list[dict] = []
    for subset in SUBSETS_FOR_RHO:
        for label, lname in [(1, "pos"), (0, "neg")]:
            n = int(
                (
                    (long_df["subset"] == subset)
                    & (long_df["label"] == label)
                    & (long_df["model"] == MODELS[0])
                    & long_df["phyloP"].notna()
                ).sum()
            )
            if n < 10:
                continue
            for model in MODELS:
                s = long_df[
                    (long_df["model"] == model)
                    & (long_df["subset"] == subset)
                    & (long_df["label"] == label)
                    & long_df["phyloP"].notna()
                ]
                rho = float(spearmanr(s["phyloP"], s["minus_llr"]).statistic)
                rows.append(
                    {
                        "subset": subset,
                        "label": lname,
                        "model": model,
                        "n": n,
                        "rho_phyloP_score": rho,
                    }
                )
    return pd.DataFrame(rows)


def plot_af_threshold_scale_curves(af_df: pd.DataFrame, out_dir: Path) -> None:
    """3-panel: raw AUPRC, AUPRC-prev, MRR — each vs AF threshold, per model."""
    df = af_df[af_df["af_thr"].notna()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for model in MODELS:
        s = df[df["model"] == model].sort_values("af_thr")
        label = model.replace("evo2_", "")
        axes[0].errorbar(
            s["af_thr"],
            s["auprc"],
            yerr=s["auprc_se"],
            marker="o",
            color=MODEL_COLORS[model],
            label=label,
            capsize=3,
        )
        axes[1].errorbar(
            s["af_thr"],
            s["auprc_minus_prev"],
            yerr=s["auprc_se"],
            marker="o",
            color=MODEL_COLORS[model],
            label=label,
            capsize=3,
        )
        axes[2].errorbar(
            s["af_thr"],
            s["mrr"],
            yerr=s["mrr_se"],
            marker="o",
            color=MODEL_COLORS[model],
            label=label,
            capsize=3,
        )
    titles = ["Raw AUPRC", "AUPRC − prevalence (gain over chance)", "MRR-within-group"]
    ylabels = ["AUPRC", "AUPRC − prev", "MRR"]
    for ax, ttl, ylab in zip(axes, titles, ylabels):
        ax.set_xscale("log")
        ax.set_xlabel("Negative-AF filter threshold")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Missense — performance vs AF-threshold filter on negatives", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_dir / "af_threshold_missense.svg", bbox_inches="tight")
    fig.savefig(out_dir / "af_threshold_missense.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phyloP_vs_score(long_df: pd.DataFrame, out_dir: Path) -> None:
    """3-panel scatter: phyloP vs minus_llr for missense, one panel per model."""
    miss = long_df[long_df["subset"] == "missense_variant"].dropna(
        subset=["phyloP", "AF"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    sc = None
    for ax, model in zip(axes, MODELS):
        s = miss[miss["model"] == model]
        neg = s[s["label"] == 0]
        pos = s[s["label"] == 1]
        sc = ax.scatter(
            neg["phyloP"],
            neg["minus_llr"],
            c=np.log10(neg["AF"].clip(lower=1e-6)),
            cmap="viridis",
            s=5,
            alpha=0.4,
            vmin=-3.5,
            vmax=-1,
            rasterized=True,
        )
        ax.scatter(
            pos["phyloP"],
            pos["minus_llr"],
            c="red",
            s=20,
            alpha=0.7,
            marker="x",
            linewidths=1,
            label=f"{len(pos)} positives",
        )
        rho_n = float(spearmanr(neg["phyloP"], neg["minus_llr"]).statistic)
        rho_p = float(spearmanr(pos["phyloP"], pos["minus_llr"]).statistic)
        ax.set_title(
            f"{model.replace('evo2_', '')}\n"
            f"ρ(phyloP, score): neg={rho_n:+.3f}, pos={rho_p:+.3f}",
            fontsize=11,
        )
        ax.set_xlabel("phyloP_241m (positive = conserved)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="grey", linewidth=0.5)
    axes[0].set_ylabel("Evo 2 minus_llr (positive = deleterious-looking)", fontsize=10)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02, aspect=30)
    cbar.set_label("log10(AF) (negatives only)", fontsize=10)
    fig.suptitle(
        "Missense — Evo 2 score vs phyloP_241m across model scale\n"
        "Negatives (dots, color = log AF) and positives (red ×). "
        "ρ_neg grows with scale: 40B is more phyloP-driven than 1B.",
        fontsize=11,
    )
    fig.savefig(out_dir / "phyloP_vs_score_missense.svg", bbox_inches="tight")
    fig.savefig(out_dir / "phyloP_vs_score_missense.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("Loading predictions + AF + phyloP_241m…")
    long_df = load_annotated()
    n_missing_phyloP = int(long_df["phyloP"].isna().sum())
    print(
        f"  merged: {len(long_df)} rows; phyloP missing in {n_missing_phyloP}/{len(long_df)} rows"
    )

    print("AF-threshold filter sweep on missense…")
    af_df = af_threshold_sweep(long_df, subset="missense_variant")
    af_df.to_parquet(OUT_DIR / "af_threshold_metrics_missense.parquet", index=False)
    print(af_df.to_string(index=False))

    print("Spearman(phyloP_241m, score) per (subset, label, model)…")
    rho_df = phyloP_spearman(long_df)
    rho_df.to_parquet(OUT_DIR / "phyloP_score_spearman.parquet", index=False)
    print(rho_df.to_string(index=False))

    print("Plotting AF-threshold scale curves…")
    plot_af_threshold_scale_curves(af_df, OUT_DIR)

    print("Plotting phyloP vs minus_llr scatter…")
    plot_phyloP_vs_score(long_df, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
