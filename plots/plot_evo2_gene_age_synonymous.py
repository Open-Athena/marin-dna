"""Gene-age + synonymous analysis for #203 fifth iteration.

Tests two hypotheses from the issue thread:
1. Older genes (more represented in genomic LM training corpora — see
   thestacks.org/publications/idea-phylogenies-bfms Fig 3) get "memorized"
   more; bigger Evo 2 may overfit to old-gene patterns and over-score
   benign variants there. Compare to GPN-Star to see if the effect is
   Evo-2-specific or general to likelihood-based variant scoring.
2. Synonymous variants are usually neutral but sit in CDS regions with
   high sequence conservation — prone to LM over-prediction of
   deleteriousness from "surroundings".

Data sources:
- Liebeskind 2016 consensus gene ages (Zenodo 51708) — modeAge per
  UniProt accession; mapped via mygene.info ENSG → UniProt.
- Per-variant predictions: Evo 2 + GPN-Star v2 gists (see issues #131
  and #145).
- phyloP_241m from S3 conservation_eval.
- HF mendelian dataset v2 (PR #194 revision).

Outputs all under `plots/output/evo2_scale_comparison/`:
- `ensg_to_age.parquet` — ENSG → UniProt → modeAge → numeric MYA
- `gene_age_correlation_missense.parquet` — Spearman(age, mean score)
- `missense_auprc_by_age_bucket.parquet` — per (model, age bucket) AUPRC
- `gene_age_top20_fps_per_model.parquet` — per-model top-20 FP gene-age stats
- `gene_age_per_variant_spearman.parquet`
- `synonymous_per_model.parquet`
- `top{20,15}_FNs_40b_synonymous.parquet`
- `missense_auprc_by_gene_age.{svg,png}`
- `synonymous_summary.{svg,png}`
- `gene_age_top20_fps_per_model.{svg,png}`

Usage:
    uv run python plots/plot_evo2_gene_age_synonymous.py
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from scipy.stats import mannwhitneyu, spearmanr

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
GENE_AGES_ZIP = "https://zenodo.org/api/records/51708/files/Gene-Ages-v1.0.zip/content"
GENE_AGES_PATH_IN_ZIP = "marcottelab-Gene-Ages-fee8d00/Main/main_HUMAN.csv"
OUT_DIR = Path(__file__).parent / "output" / "evo2_scale_comparison"

# Approximate divergence times (MYA) per Liebeskind 2016 / TimeTree.
# Rough but adequate for ordering buckets and per-variant ages.
MODE_AGE_MYA: dict[str, int] = {
    "Cellular_organisms": 4290,
    "Euk_Archaea": 3000,
    "Euk+Bac": 3500,
    "Eukaryota": 1962,
    "Opisthokonta": 1105,
    "Eumetazoa": 824,
    "Vertebrata": 615,
    "Mammalia": 320,
}

MODELS = (
    "evo2_1b_base",
    "evo2_7b",
    "evo2_40b",
    "GPN-Star-V-cLLR",
    "GPN-Star-M-cLLR",
    "GPN-Star-P-cLLR",
)
MODEL_COLORS = {
    "evo2_1b_base": "#3b1a64",
    "evo2_7b": "#3a6fa0",
    "evo2_40b": "#e8d840",
    "GPN-Star-V-cLLR": "#1e7a1e",
    "GPN-Star-M-cLLR": "#a3a30b",
    "GPN-Star-P-cLLR": "#c44d4d",
}


def load_gene_ages() -> pd.DataFrame:
    """Liebeskind 2016 consensus gene ages keyed by UniProt accession,
    with modeAge bucket + approximate MYA."""
    r = requests.get(GENE_AGES_ZIP, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        with zf.open(GENE_AGES_PATH_IN_ZIP) as f:
            ages = pd.read_csv(f, index_col=0)
    ages["age_mya"] = ages["modeAge"].map(MODE_AGE_MYA)
    return ages[["modeAge", "age_mya"]]


def map_ensg_to_uniprot(ensgs: list[str], batch: int = 500) -> dict[str, str]:
    """Bulk ENSG → UniProt via mygene.info (Swiss-Prot only)."""
    out: dict[str, str] = {}
    for i in range(0, len(ensgs), batch):
        chunk = ensgs[i : i + batch]
        r = requests.post(
            "https://mygene.info/v3/query",
            data={
                "q": ",".join(chunk),
                "scopes": "ensembl.gene",
                "fields": "uniprot.Swiss-Prot",
                "species": "human",
            },
            timeout=60,
        )
        if r.status_code != 200:
            print(f"  batch {i}: HTTP {r.status_code} — skipping")
            continue
        for hit in r.json():
            q = hit.get("query")
            u = (hit.get("uniprot") or {}).get("Swiss-Prot")
            if isinstance(u, list):
                u = u[0] if u else None
            if u and q and q not in out:
                out[q] = u
        time.sleep(0.3)
    return out


def load_wide() -> pd.DataFrame:
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
            "exon_closest_pc_gene_id",
        ]
    ]
    ann["chrom"] = ann["chrom"].astype(str)
    phyloP = pd.read_parquet(PHYLOP_S3)[["chrom", "pos", "ref", "alt", "score"]].rename(
        columns={"score": "phyloP"}
    )
    phyloP["chrom"] = phyloP["chrom"].astype(str)
    wide = ann.merge(phyloP, on=["chrom", "pos", "ref", "alt"], how="left")

    print("Loading gene-age table + mapping ENSGs…")
    ages = load_gene_ages()
    unique_ensgs = ann["exon_closest_pc_gene_id"].dropna().unique().tolist()
    ensg_to_uni = map_ensg_to_uniprot(unique_ensgs)
    ensg_age = pd.DataFrame(
        [{"exon_closest_pc_gene_id": k, "uniprot": v} for k, v in ensg_to_uni.items()]
    ).merge(
        ages.reset_index().rename(columns={"index": "uniprot"}),
        on="uniprot",
        how="left",
    )
    ensg_age.to_parquet(OUT_DIR / "ensg_to_age.parquet", index=False)
    print(
        f"  mapped {len(ensg_to_uni)} / {len(unique_ensgs)} ENSGs, "
        f"{ensg_age['age_mya'].notna().sum()} have numeric age"
    )
    wide = wide.merge(
        ensg_age[["exon_closest_pc_gene_id", "modeAge", "age_mya"]],
        on="exon_closest_pc_gene_id",
        how="left",
    )
    for m_short in ("1b_base", "7b", "40b"):
        d = pd.read_parquet(EVO_GISTS[f"evo2_{m_short}"])[
            ["chrom", "pos", "ref", "alt", "minus_llr"]
        ].rename(columns={"minus_llr": f"evo2_{m_short}"})
        d["chrom"] = d["chrom"].astype(str)
        wide = wide.merge(d, on=["chrom", "pos", "ref", "alt"], how="left")
    for v, url in GPN_GISTS.items():
        g = pd.read_parquet(url)
        g["chrom"] = g["chrom"].astype(str)
        g = g[g["split"] == "train"].copy()
        g[f"GPN-Star-{v}-cLLR"] = -g["llr_calibrated"]
        wide = wide.merge(
            g[["chrom", "pos", "ref", "alt", f"GPN-Star-{v}-cLLR"]],
            on=["chrom", "pos", "ref", "alt"],
            how="left",
        )
    return wide


def missense_auprc_by_age_bucket(wide: pd.DataFrame) -> pd.DataFrame:
    miss = wide[(wide["subset"] == "missense_variant") & wide["age_mya"].notna()]
    rows: list[dict] = []
    for bucket in MODE_AGE_MYA:
        sub = miss[miss["modeAge"] == bucket]
        if sub["match_group"].nunique() < 30:
            continue
        for model in MODELS:
            ss = sub.dropna(subset=[model, "label", "match_group"])
            counts = ss.groupby("match_group")["label"].agg(["sum", "count"])
            bad = (
                counts[(counts["sum"] != 1) | (counts["count"] < 2)]
                .reset_index()["match_group"]
                .unique()
            )
            ss = ss[~ss["match_group"].isin(bad)]
            if ss["match_group"].nunique() < 10:
                continue
            r = auprc_with_bootstrap_se(
                ss["label"], ss[model], ss["match_group"], n_bootstrap=200, rng=0
            )
            rows.append(
                {
                    "age_bucket": bucket,
                    "model": model,
                    "auprc": r["value"],
                    "auprc_se": r["se"],
                    "n_groups": r["n_groups"],
                }
            )
    return pd.DataFrame(rows)


def top_fp_gene_age_stats(wide: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """For each model, age distribution of its top-N missense FPs (negatives)
    vs all-negatives baseline."""
    miss_neg = wide[
        (wide["subset"] == "missense_variant")
        & (wide["label"] == False)
        & wide["age_mya"].notna()
    ]
    rows: list[dict] = []
    for model in MODELS:
        top = miss_neg.nlargest(top_n, model)
        med = float(top["age_mya"].median())
        mean = float(top["age_mya"].mean())
        pval = float(
            mannwhitneyu(
                top["age_mya"], miss_neg["age_mya"], alternative="greater"
            ).pvalue
        )
        rows.append(
            {
                "model": model,
                "top_n": top_n,
                "top_median_age_mya": med,
                "top_mean_age_mya": mean,
                "pval_older_than_baseline": pval,
                "baseline_median_age_mya": float(miss_neg["age_mya"].median()),
            }
        )
    return pd.DataFrame(rows)


def synonymous_per_model(wide: pd.DataFrame) -> pd.DataFrame:
    syn = wide[wide["subset"] == "synonymous_variant"]
    rows: list[dict] = []
    for model in MODELS:
        ss = syn.dropna(subset=[model])
        r = auprc_with_bootstrap_se(
            ss["label"], ss[model], ss["match_group"], n_bootstrap=300, rng=0
        )
        sn = ss[(ss["label"] == False) & ss["phyloP"].notna()]
        sp = ss[(ss["label"] == True) & ss["phyloP"].notna()]
        rho_n = float(spearmanr(sn["phyloP"], sn[model]).statistic)
        rho_p = float(spearmanr(sp["phyloP"], sp[model]).statistic)
        rows.append(
            {
                "model": model,
                "auprc": r["value"],
                "auprc_se": r["se"],
                "rho_phyloP_neg": rho_n,
                "rho_phyloP_pos": rho_p,
            }
        )
    return pd.DataFrame(rows)


def plot_age_bucket_auprc(df: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "Vertebrata",
        "Eumetazoa",
        "Opisthokonta",
        "Eukaryota",
        "Cellular_organisms",
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    n_b = len(order)
    n_m = len(MODELS)
    w = 0.85 / n_m
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == m].set_index("age_bucket").reindex(order)
        pos = np.arange(n_b) + i * w - 0.42
        ax.bar(
            pos,
            sub["auprc"],
            width=w,
            label=m,
            color=MODEL_COLORS[m],
            edgecolor="black",
            linewidth=0.4,
            yerr=sub["auprc_se"],
            capsize=2,
            alpha=0.95,
        )
    labels = [f"{b}\n(~{MODE_AGE_MYA[b]} MYA)" for b in order]
    ax.set_xticks(np.arange(n_b))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Gene age bucket (Liebeskind modeAge, approx MYA)")
    ax.set_ylabel("Missense AUPRC")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Missense AUPRC stratified by gene age", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "missense_auprc_by_gene_age.svg", bbox_inches="tight")
    fig.savefig(
        out_dir / "missense_auprc_by_gene_age.png", dpi=140, bbox_inches="tight"
    )
    plt.close(fig)


def plot_top_fp_age_boxes(wide: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    miss_neg = wide[
        (wide["subset"] == "missense_variant")
        & (wide["label"] == False)
        & wide["age_mya"].notna()
    ]
    data_list = [miss_neg.nlargest(top_n, m)["age_mya"] for m in MODELS] + [
        miss_neg["age_mya"]
    ]
    labels = [m.replace("evo2_", "").replace("-cLLR", "") for m in MODELS] + [
        "all neg\n(baseline)"
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, m in zip(bp["boxes"][:-1], MODELS):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.75)
    bp["boxes"][-1].set_facecolor("#cccccc")
    bp["boxes"][-1].set_alpha(0.75)
    ax.axhline(
        miss_neg["age_mya"].median(),
        color="grey",
        linestyle="--",
        linewidth=0.8,
        label=f"baseline median ({miss_neg['age_mya'].median():.0f} MYA)",
    )
    ax.set_ylabel("Gene age (MYA)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        f"Are top-{top_n} missense FPs concentrated in older genes?\n"
        f"Box = top-{top_n} highest-scoring negatives per model",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "gene_age_top20_fps_per_model.svg", bbox_inches="tight")
    fig.savefig(
        out_dir / "gene_age_top20_fps_per_model.png", dpi=140, bbox_inches="tight"
    )
    plt.close(fig)


def plot_synonymous(syn_df: pd.DataFrame, out_dir: Path) -> None:
    syn = syn_df.set_index("model").reindex(list(MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(syn))
    axes[0].bar(
        x,
        syn["auprc"],
        yerr=syn["auprc_se"],
        capsize=3,
        color=[MODEL_COLORS[m] for m in MODELS],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(MODELS), rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Synonymous AUPRC")
    axes[0].set_title("Synonymous AUPRC (n_groups=46)", fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(
        x - 0.2,
        syn["rho_phyloP_neg"],
        width=0.4,
        color="#4477AA",
        label="negatives",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].bar(
        x + 0.2,
        syn["rho_phyloP_pos"],
        width=0.4,
        color="#EE6677",
        label="positives",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(list(MODELS), rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Spearman(phyloP_241m, minus_llr)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("Synonymous — phyloP correlation per model", fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "Synonymous variants — generally neutral but in conserved coding contexts",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "synonymous_summary.svg", bbox_inches="tight")
    fig.savefig(out_dir / "synonymous_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    wide = load_wide()
    print(
        f"wide: {len(wide)} rows × {len(wide.columns)} cols; "
        f"missense age cov = {wide[wide['subset'] == 'missense_variant']['age_mya'].notna().mean():.3f}"
    )

    print("Missense AUPRC by gene age bucket…")
    age_auprc = missense_auprc_by_age_bucket(wide)
    age_auprc.to_parquet(OUT_DIR / "missense_auprc_by_age_bucket.parquet", index=False)
    print(age_auprc.to_string(index=False))

    print("Top-20 FP gene-age stats per model…")
    top_fp_stats = top_fp_gene_age_stats(wide, top_n=20)
    top_fp_stats.to_parquet(
        OUT_DIR / "gene_age_top20_fps_per_model.parquet", index=False
    )
    print(top_fp_stats.to_string(index=False))

    print("Synonymous per-model…")
    syn = synonymous_per_model(wide)
    syn.to_parquet(OUT_DIR / "synonymous_per_model.parquet", index=False)
    print(syn.to_string(index=False))

    print("Plotting…")
    plot_age_bucket_auprc(age_auprc, OUT_DIR)
    plot_top_fp_age_boxes(wide, OUT_DIR, top_n=20)
    plot_synonymous(syn, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
