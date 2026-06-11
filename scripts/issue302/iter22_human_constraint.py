"""issue #302 — iteration 22: are the confidently-over-called benigns in genes/sites that are
conserved across species but TOLERANT in humans (GB's refined hypothesis — relaxed constraint
on the human lineage, not on primates)?

Forensic lookup (myvariant.info / dbNSFP, hg38) on the top confident FPs vs true pathogenics
(TP) vs ordinary benigns (B0), all in the high-LLR confident set:
  - ancestral_allele: is the tolerated ALT the ancestral state (human REF derived)? -> tests
    the "human reverted an ancestral change" version of the hypothesis.
  - AlphaMissense (.pred) / REVEL: do human-aware supervised VEPs call them benign? (they use
    human+primate population data the gLM lacks)
  - gnomAD v2.1.1 missense constraint (oe_mis, LOEUF): are the FP genes more missense-tolerant
    in humans than the true-pathogenic genes? -> the gene-level "relaxed human constraint" test.

Reads scores from S3; queries myvariant.info; caches the gnomAD constraint table. CPU.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter22_human_constraint.py
"""

from __future__ import annotations

import collections
import gzip
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests

S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
CONSTRAINT = "scratch/issue302/gnomad_constraint.tsv"
GNOMAD_URL = "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
N = 120


def _first(x):
    return (x[0] if x else None) if isinstance(x, list) else x


def _fmax(x):
    o: list[float] = []

    def w(v):
        if isinstance(v, (int, float)):
            o.append(float(v))
        elif isinstance(v, list):
            [w(e) for e in v]
        elif isinstance(v, dict):
            [w(e) for e in v.values()]

    w(x)
    return max(o) if o else None


def _ampred(db):
    p = ((db.get("alphamissense") or {}) or {}).get("pred")
    if not p:
        return None
    p = p if isinstance(p, list) else [p]
    return collections.Counter(p).most_common(1)[0][0]


def main() -> None:
    m = (
        pl.read_parquet(S)
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(
            pl.col("chrom").cast(str),
            pl.col("AF").fill_null(0.0),
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
        )
    )
    pm = m.filter(pl.col("label") == 1)["mll"].median()
    sets = {
        "FP": m.filter((pl.col("label") == 0) & (pl.col("mll") >= pm))
        .sort("mll", descending=True)
        .head(N),
        "TP": m.filter((pl.col("label") == 1) & (pl.col("mll") >= pm))
        .sort("mll", descending=True)
        .head(N),
        "B0": m.filter((pl.col("label") == 0) & (pl.col("mll") < pm)).head(N),
    }
    allrows = [
        {**r, "grp": g} for g, df in sets.items() for r in df.iter_rows(named=True)
    ]
    ids = [f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}" for r in allrows]
    resp = requests.post(
        "https://myvariant.info/v1/variant",
        data={
            "ids": ",".join(ids),
            "fields": "dbnsfp.genename,dbnsfp.ancestral_allele,dbnsfp.alphamissense.pred,dbnsfp.revel.score",
            "assembly": "hg38",
        },
        timeout=180,
    ).json()
    byid = {d.get("query"): d for d in resp}

    if not os.path.exists(CONSTRAINT):
        open(CONSTRAINT, "w").write(
            gzip.decompress(requests.get(GNOMAD_URL, timeout=240).content).decode()
        )
    con = pl.read_csv(
        CONSTRAINT, separator="\t", null_values=["NA"], infer_schema_length=0
    ).with_columns(
        pl.col("oe_mis").cast(pl.Float64, strict=False),
        pl.col("oe_lof_upper").cast(pl.Float64, strict=False),
    )
    cmap = {
        r["gene"]: r
        for r in con.select(["gene", "oe_mis", "oe_lof_upper"]).iter_rows(named=True)
    }

    rows, genes, fp_table = [], {g: [] for g in sets}, []
    for g in ("TP", "FP", "B0"):
        n = aa = ab = ap = rh = 0
        for r in [x for x in allrows if x["grp"] == g]:
            d = byid.get(f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}", {})
            if not d or d.get("notfound"):
                continue
            n += 1
            db = d.get("dbnsfp", {}) or {}
            anc = db.get("ancestral_allele")
            aa += bool(anc) and str(anc).upper() == r["alt"].upper()
            pr = _ampred(db)
            ab += pr == "B"
            ap += pr == "P"
            rv = _fmax(db.get("revel"))
            rh += rv is not None and rv > 0.5
            gn = _first(db.get("genename"))
            if gn:
                genes[g].append(gn)
            if g == "FP":
                fp_table.append(
                    {
                        "v": f"{r['chrom']}:{r['pos']}{r['ref']}>{r['alt']}",
                        "AF": round(r["AF"], 4),
                        "gene": gn,
                        "AM": pr,
                        "revel": rv,
                        "oe_mis": (cmap.get(gn) or {}).get("oe_mis"),
                    }
                )
        rows.append(
            {
                "grp": g,
                "n": n,
                "anc_alt": aa / n,
                "am_benign": ab / n,
                "am_patho": ap / n,
                "revel_hi": rh / n,
            }
        )
        print(
            f"{g}: n={n} anc=alt={aa / n:.0%} AM_B={ab / n:.0%} AM_P={ap / n:.0%} REVEL>.5={rh / n:.0%}"
        )

    oe = {
        g: [
            cmap[x]["oe_mis"]
            for x in set(genes[g])
            if x in cmap and cmap[x]["oe_mis"] is not None
        ]
        for g in sets
    }
    loeuf = {
        g: [
            cmap[x]["oe_lof_upper"]
            for x in set(genes[g])
            if x in cmap and cmap[x]["oe_lof_upper"] is not None
        ]
        for g in sets
    }
    for g in ("TP", "FP", "B0"):
        print(
            f"{g} genes: median oe_mis={np.median(oe[g]):.2f}  median LOEUF={np.median(loeuf[g]):.2f}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet("scratch/issue302/human_constraint.parquet")
    open("scratch/issue302/fp_table.json", "w").write(
        json.dumps(sorted(fp_table, key=lambda x: -(x["oe_mis"] or 0))[:20], indent=1)
    )

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    sig = ["anc_alt", "am_benign", "am_patho", "revel_hi"]
    labs = [
        "ALT=ancestral\n(human-derived?)",
        "AlphaMissense\n=benign",
        "AlphaMissense\n=pathogenic",
        "REVEL>0.5",
    ]
    rr = {x["grp"]: x for x in rows}
    x = np.arange(len(sig))
    for k, g in enumerate(["TP", "FP"]):
        ax[0].bar(
            x + (k - 0.5) * 0.4,
            [rr[g][s] * 100 for s in sig],
            0.4,
            label=f"{g}",
            color=["tab:red", "tab:orange"][k],
        )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labs, fontsize=8)
    ax[0].set_ylabel("% of variants")
    ax[0].set_title(
        "Human-aware VEPs are NOT fooled by the FPs\n(AlphaMissense/REVEL call most benign); and they're not ancestral-reversions"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    parts = ax[1].violinplot([oe[g] for g in ("TP", "FP", "B0")], showmedians=True)
    for pc, c in zip(parts["bodies"], ["tab:red", "tab:orange", "tab:blue"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax[1].axhline(
        1.0, color="gray", ls=":", lw=1, label="oe_mis=1 (no human constraint)"
    )
    ax[1].set_xticks([1, 2, 3])
    ax[1].set_xticklabels(
        [
            f"TP\n(med {np.median(oe['TP']):.2f})",
            f"FP\n(med {np.median(oe['FP']):.2f})",
            f"B0\n(med {np.median(oe['B0']):.2f})",
        ],
        fontsize=9,
    )
    ax[1].set_ylabel("gnomAD missense oe_mis (obs/exp; higher = more human-tolerant)")
    ax[1].set_title(
        "The FP genes are MORE missense-tolerant in humans\nthan the true-pathogenic genes"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "The confident FPs sit in genes conserved across species but RELAXED in humans — and human/primate-aware VEPs catch them",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "human_constraint.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "human_constraint.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'human_constraint'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/human_constraint.parquet",
        str(OUT / "human_constraint.png"),
        str(OUT / "human_constraint.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
