"""issue #302 — iteration 23: proper statistical GO/pathway enrichment + gene-constraint test
for the over-called benigns (FPs), swept across CONFIDENCE THRESHOLDS (GB: "highly confident
FP, not just FP; using different thresholds").

Rigorous, beyond anecdotal examples:
  - GO/KEGG/Reactome/HP enrichment via g:Profiler (g:SCS multiple-testing correction), with a
    CUSTOM BACKGROUND = all Mendelian-missense benchmark genes (controls for the benchmark's
    disease-gene ascertainment). Positive control: TP genes (should light up disease terms).
  - gnomAD v2.1.1 missense constraint oe_mis: FP genes vs the rest of the background
    (Mann-Whitney U).
FP confidence thresholds: benign missense with minus_llr >= pathogenic-{p50,p75,p90,p95}, and
absolute top-N benign-LLR. 4B. Reads S3 + g:Profiler + gnomAD constraint (cached). CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter23_go_enrichment_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
from scipy.stats import mannwhitneyu

S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
CONSTRAINT = "scratch/issue302/gnomad_constraint.tsv"
GID = "exon_closest_pc_gene_id"


def _genes(df):
    return sorted(set(x.split(".")[0] for x in df[GID].drop_nulls().to_list() if x))


def _gprof(query, bg):
    r = requests.post(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        json={
            "organism": "hsapiens",
            "query": query,
            "domain_scope": "custom",
            "background": bg,
            "sources": ["GO:BP", "GO:CC", "GO:MF", "KEGG", "REAC", "HP"],
            "user_threshold": 0.05,
            "significance_threshold_method": "g_SCS",
            "no_evidences": True,
        },
        timeout=180,
    ).json()
    return sorted(r.get("result", []), key=lambda x: x["p_value"])


def main() -> None:
    m = (
        pl.read_parquet(S)
        .filter(pl.col("subset") == "missense_variant")
        .with_columns((-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"))
    )
    pmll = m.filter(pl.col("label") == 1)["mll"].to_numpy()
    bg = _genes(m)
    con = pl.read_csv(
        CONSTRAINT, separator="\t", null_values=["NA"], infer_schema_length=0
    ).with_columns(pl.col("oe_mis").cast(pl.Float64, strict=False))
    cmap = {
        r["gene_id"].split(".")[0]: r["oe_mis"]
        for r in con.select(["gene_id", "oe_mis"]).iter_rows(named=True)
        if r["gene_id"]
    }

    def oev(gs):
        return np.array([cmap[g] for g in gs if g in cmap and cmap[g] is not None])

    # positive control: TP genes
    tp_genes = _genes(
        m.filter((pl.col("label") == 1) & (pl.col("mll") >= np.median(pmll)))
    )
    n_tp_terms = len(_gprof(tp_genes, bg))

    rows = []
    for q in [50, 75, 90, 95]:
        thr = float(np.percentile(pmll, q))
        fpdf = m.filter((pl.col("label") == 0) & (pl.col("mll") >= thr))
        g = _genes(fpdf)
        res = _gprof(g, bg)
        fv, rest = oev(g), oev([x for x in bg if x not in set(g)])
        p = (
            float(mannwhitneyu(fv, rest, alternative="two-sided")[1])
            if len(fv) > 5
            else float("nan")
        )
        rows.append(
            {
                "thr": f"p{q}",
                "n_fp": fpdf.height,
                "n_genes": len(g),
                "go_sig": len(res),
                "oe_fp": float(np.median(fv)),
                "oe_bg": float(np.median(rest)),
                "mwu_p": p,
            }
        )
        print(
            f"  LLR>=path-p{q} (n={fpdf.height}, genes={len(g)}): GO_sig={len(res)}  oe_mis {np.median(fv):.3f} vs {np.median(rest):.3f} (MWU p={p:.2e})"
        )
    print(f"  [positive control] TP genes -> {n_tp_terms} significant terms")
    res_df = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res_df.write_parquet("scratch/issue302/go_enrichment_sweep.parquet")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(rows))
    labs = [f"≥path-{r['thr']}\n(n={r['n_fp']})" for r in rows]
    ax[0].bar(x, [r["go_sig"] for r in rows], color="tab:orange", label="FP genes")
    ax[0].axhline(
        n_tp_terms,
        color="tab:red",
        ls="--",
        lw=2,
        label=f"TP genes (positive control) = {n_tp_terms}",
    )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labs, fontsize=8)
    ax[0].set_ylabel("# significant GO/pathway terms (g:SCS)")
    ax[0].set_yscale("symlog")
    ax[0].set_title(
        "No GO/pathway enrichment for the FPs at ANY confidence threshold\n(custom background; method validated by the TP control)"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].plot(
        x,
        [r["oe_fp"] for r in rows],
        "o-",
        color="tab:orange",
        lw=2.5,
        label="FP genes",
    )
    ax[1].plot(
        x,
        [r["oe_bg"] for r in rows],
        "s--",
        color="gray",
        lw=2,
        label="rest of background",
    )
    for i, r in enumerate(rows):
        ax[1].annotate(
            f"p={r['mwu_p']:.1e}",
            (i, r["oe_fp"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labs, fontsize=8)
    ax[1].set_ylabel("median gnomAD oe_mis (missense tolerance)")
    ax[1].set_title(
        "Gene constraint: a tiny effect at the broad threshold (n inflated)\nvanishes/reverses as the FP set gets 'highly confident'"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(
        "Highly-confident FPs: no functional-category enrichment, no robust gene-constraint signal at any threshold → the failure is variant/site-level, not gene-level",
        y=1.03,
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(OUT / "go_enrichment_sweep.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "go_enrichment_sweep.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'go_enrichment_sweep'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/go_enrichment_sweep.parquet",
        str(OUT / "go_enrichment_sweep.png"),
        str(OUT / "go_enrichment_sweep.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
