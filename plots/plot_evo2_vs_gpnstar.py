"""GPN-Star vs Evo 2 comparison + dbSNP/ClinVar lookup for #203.

Builds a head-to-head table on the same Mendelian k=9 dataset (PR #194):
- AUPRC per (model, subset) for 3 Evo 2 sizes (LLR) + 3 GPN-Star variants
  (LLR + cLLR), with cluster-bootstrap SE
- Spearman(phyloP_241m, minus_llr) per (model, label) on missense — shows
  GPN-Star is MORE phyloP-driven than Evo 2 40B yet achieves better
  AUPRC, so phyloP-correlation alone isn't the failure mode
- Top-20 false positives + top-20 false negatives for Evo 2 40B on
  missense, with rsID / ClinVar / CADD via myvariant.info

Inputs:
- Evo 2 predictions: gists from #131 (2026-05-20 comment), train split
- GPN-Star predictions: gists from #145 (2026-05-19 v2 comment), train split
- HF dataset: bolinas-dna/evals_mendelian_traits @ 4aed58e
- phyloP_241m: S3 conservation_eval pipeline output

Usage:
    uv run python plots/plot_evo2_vs_gpnstar.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from scipy.stats import spearmanr

from bolinas.pipelines.evals.metrics import auprc_with_bootstrap_se

EVO_GISTS: dict[str, str] = {
    "evo2_1b_base": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/b6c254849a71ca0783a24218a4fe9037e887e8f7/evo2_1b_base_train.parquet",
    "evo2_7b": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/e72d3d2e14955a670b8229dc8d525a69ea88c05c/evo2_7b_train.parquet",
    "evo2_40b": "https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/2b425e759811c201ca806ae4c8733fd7732220a6/evo2_40b_train.parquet",
}
GPN_GISTS: dict[str, str] = {
    v: f"https://gist.githubusercontent.com/gonzalobenegas/db282f89aa00244fbb7437dce0f069ef/raw/02484d50d9bfd80337e313652b26f98a9362b6b1/bolinas_mendelian_traits_GPN-Star-{v}.parquet"
    for v in ("V", "M", "P")
}
DATASET_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
PHYLOP_S3 = (
    "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits/"
    "phyloP_241m_train.parquet"
)
OUT_DIR = Path(__file__).parent / "output" / "evo2_scale_comparison"

EVO_MODELS = ("evo2_1b_base", "evo2_7b", "evo2_40b")
GPN_MODELS = ("GPN-Star-V", "GPN-Star-M", "GPN-Star-P")
ALL_MODELS = (
    "evo2_1b_base",
    "evo2_7b",
    "evo2_40b",
    "GPN-Star-V",
    "GPN-Star-V-cLLR",
    "GPN-Star-M",
    "GPN-Star-M-cLLR",
    "GPN-Star-P",
    "GPN-Star-P-cLLR",
)
SUBSETS = (
    "missense_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "distal",
)
MODEL_COLORS = {
    "evo2_1b_base": "#3b1a64",
    "evo2_7b": "#3a6fa0",
    "evo2_40b": "#e8d840",
    "GPN-Star-V": "#1e7a1e",
    "GPN-Star-V-cLLR": "#1e7a1e",
    "GPN-Star-M": "#a3a30b",
    "GPN-Star-M-cLLR": "#a3a30b",
    "GPN-Star-P": "#c44d4d",
    "GPN-Star-P-cLLR": "#c44d4d",
}
HATCHES = {"GPN-Star-V-cLLR": "//", "GPN-Star-M-cLLR": "//", "GPN-Star-P-cLLR": "//"}


def load_wide() -> pd.DataFrame:
    """Long table: one row per variant, with score columns per model."""
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
            "label",
            "subset",
            "match_group",
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
    wide = ann.merge(phyloP, on=["chrom", "pos", "ref", "alt"], how="left")
    for model, url in EVO_GISTS.items():
        d = pd.read_parquet(url)[["chrom", "pos", "ref", "alt", "minus_llr"]].rename(
            columns={"minus_llr": model}
        )
        d["chrom"] = d["chrom"].astype(str)
        wide = wide.merge(d, on=["chrom", "pos", "ref", "alt"], how="left")
    for v, url in GPN_GISTS.items():
        g = pd.read_parquet(url)
        g["chrom"] = g["chrom"].astype(str)
        g = g[g["split"] == "train"].copy()
        g[f"GPN-Star-{v}"] = -g["llr"]
        g[f"GPN-Star-{v}-cLLR"] = -g["llr_calibrated"]
        wide = wide.merge(
            g[["chrom", "pos", "ref", "alt", f"GPN-Star-{v}", f"GPN-Star-{v}-cLLR"]],
            on=["chrom", "pos", "ref", "alt"],
            how="left",
        )
    return wide


def per_subset_auprc(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, subset) AUPRC + cluster-bootstrap SE."""
    rows: list[dict] = []
    for model in ALL_MODELS:
        for subset in SUBSETS:
            s = wide[wide["subset"] == subset][["label", model, "match_group"]].dropna()
            if len(s) < 10:
                continue
            r = auprc_with_bootstrap_se(
                s["label"], s[model], s["match_group"], n_bootstrap=200, rng=0
            )
            rows.append(
                {
                    "model": model,
                    "subset": subset,
                    "auprc": r["value"],
                    "auprc_se": r["se"],
                    "n_groups": r["n_groups"],
                }
            )
    return pd.DataFrame(rows)


def phyloP_spearman_missense(wide: pd.DataFrame) -> pd.DataFrame:
    """Spearman(phyloP_241m, score) per model on missense, split by label."""
    miss = wide[(wide["subset"] == "missense_variant") & wide["phyloP"].notna()]
    rows: list[dict] = []
    for model in ALL_MODELS:
        m = miss.dropna(subset=[model])
        rho_n = float(
            spearmanr(m[m["label"] == 0]["phyloP"], m[m["label"] == 0][model]).statistic
        )
        rho_p = float(
            spearmanr(m[m["label"] == 1]["phyloP"], m[m["label"] == 1][model]).statistic
        )
        rows.append(
            {
                "model": model,
                "subset": "missense_variant",
                "rho_neg": rho_n,
                "rho_pos": rho_p,
            }
        )
    return pd.DataFrame(rows)


def myvariant_lookup(hgvs_id: str, retries: int = 2) -> dict | None:
    url = (
        f"https://myvariant.info/v1/variant/{hgvs_id}?assembly=hg38&"
        "fields=dbsnp.rsid,dbsnp.gene.symbol,clinvar.rcv.clinical_significance,"
        "clinvar.rcv.conditions.name,cadd.phred,cadd.consequence,"
        "gnomad_genome.af.af"
    )
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(0.5)
    return None


def extract_meta(d: dict | None) -> dict:
    if not d:
        return {
            "rsid": "",
            "gene": "",
            "clinvar": "",
            "cadd_phred": None,
            "cadd_consequence": "",
            "gnomad_af": None,
        }
    out: dict = {}
    out["rsid"] = (d.get("dbsnp") or {}).get("rsid", "") or ""
    gene = (d.get("dbsnp") or {}).get("gene", {})
    if isinstance(gene, list):
        gene = gene[0] if gene else {}
    out["gene"] = (gene or {}).get("symbol", "")
    cv = (d.get("clinvar") or {}).get("rcv")
    if isinstance(cv, list):
        cv = cv[0] if cv else None
    out["clinvar"] = (cv or {}).get("clinical_significance", "")
    cadd = d.get("cadd") or {}
    out["cadd_phred"] = cadd.get("phred")
    cons = cadd.get("consequence", "")
    if isinstance(cons, list):
        cons = "+".join(c for c in cons if c)[:30]
    out["cadd_consequence"] = cons
    gg = (d.get("gnomad_genome") or {}).get("af", {}) or {}
    out["gnomad_af"] = gg.get("af")
    return out


def annotate_top(top: pd.DataFrame) -> pd.DataFrame:
    metas: list[dict] = []
    for _, row in top.iterrows():
        h = f"chr{row['chrom']}:g.{row['pos']}{row['ref']}>{row['alt']}"
        meta = extract_meta(myvariant_lookup(h))
        meta["hgvs"] = h
        metas.append(meta)
        time.sleep(0.15)  # polite throttle on myvariant.info
    meta_df = pd.DataFrame(metas)
    out = top.reset_index(drop=True).join(
        meta_df.drop(columns=["hgvs"]).reset_index(drop=True)
    )
    out["hgvs"] = meta_df["hgvs"].values
    return out


def plot_per_subset_auprc(auprc_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    n_models = len(ALL_MODELS)
    n_sub = len(SUBSETS)
    w = 0.85 / n_models
    for i, m in enumerate(ALL_MODELS):
        sub = auprc_df[auprc_df["model"] == m].set_index("subset").reindex(SUBSETS)
        pos = np.arange(n_sub) + i * w - 0.42
        ax.bar(
            pos,
            sub["auprc"],
            width=w,
            label=m,
            color=MODEL_COLORS[m],
            edgecolor="black",
            linewidth=0.4,
            hatch=HATCHES.get(m, ""),
            alpha=0.95,
            yerr=sub["auprc_se"],
            capsize=2,
        )
    ax.set_xticks(np.arange(n_sub))
    ax.set_xticklabels(SUBSETS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 0.85)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        "Per-subset AUPRC — Evo 2 (3 sizes, LLR) vs GPN-Star (V/M/P, LLR + cLLR)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "evo2_vs_gpnstar_per_subset.svg", bbox_inches="tight")
    fig.savefig(
        out_dir / "evo2_vs_gpnstar_per_subset.png", dpi=140, bbox_inches="tight"
    )
    plt.close(fig)


def plot_phyloP_correlation(rho_df: pd.DataFrame, out_dir: Path) -> None:
    rho = rho_df.set_index("model").reindex(list(ALL_MODELS))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(rho))
    w = 0.4
    ax.bar(
        x - w / 2,
        rho["rho_neg"],
        width=w,
        color="#4477AA",
        label="negatives",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + w / 2,
        rho["rho_pos"],
        width=w,
        color="#EE6677",
        label="positives",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(ALL_MODELS), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Spearman(phyloP_241m, minus_llr)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(
        "How phyloP-driven is each model's missense score? (Spearman)", fontsize=11
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "phyloP_correlation_evo2_vs_gpnstar.svg", bbox_inches="tight")
    fig.savefig(
        out_dir / "phyloP_correlation_evo2_vs_gpnstar.png", dpi=140, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading predictions + AF + phyloP_241m…")
    wide = load_wide()
    print(f"  wide: {len(wide)} rows × {len(wide.columns)} cols")

    print("Per-subset AUPRC (cluster-bootstrap SE)…")
    auprc_df = per_subset_auprc(wide)
    auprc_df.to_parquet(
        OUT_DIR / "evo2_vs_gpnstar_auprc_per_subset.parquet", index=False
    )

    print("Spearman(phyloP_241m, score) per model on missense…")
    rho_df = phyloP_spearman_missense(wide)
    rho_df.to_parquet(
        OUT_DIR / "phyloP_correlation_evo2_vs_gpnstar.parquet", index=False
    )
    print(rho_df.to_string(index=False))

    print("dbSNP / ClinVar / CADD lookup for 40B's top-20 missense FPs and FNs…")
    miss = wide[wide["subset"] == "missense_variant"]
    top_fps = miss[miss["label"] == False].nlargest(20, "evo2_40b").copy()
    top_fns = miss[miss["label"] == True].nsmallest(20, "evo2_40b").copy()
    fp = annotate_top(top_fps)
    fn = annotate_top(top_fns)
    fp.to_parquet(OUT_DIR / "dbsnp_top20_FPs_40b_missense.parquet", index=False)
    fn.to_parquet(OUT_DIR / "dbsnp_top20_FNs_40b_missense.parquet", index=False)

    print("Plotting per-subset AUPRC bars + phyloP-correlation bars…")
    plot_per_subset_auprc(auprc_df, OUT_DIR)
    plot_phyloP_correlation(rho_df, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
