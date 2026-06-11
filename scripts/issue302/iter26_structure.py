"""issue #302 — iteration 26: are the over-called benigns at structurally TOLERANT positions
(disordered / low-confidence), the one site-level feature that could distinguish them from true
pathogenics (everything sequence-level shows FP ~ TP)? AlphaMissense — which catches the FPs
(iter22) — is structure-aware, so this is the natural test.

Per variant: map to UniProt + residue via dbNSFP (hgvsp is isoform-dependent, so we VALIDATE
each (acc, isoform-residue) against the AlphaFold structure's actual amino acid and keep only
matches), then read the residue pLDDT (AlphaFold confidence; low = disordered/flexible).
Compare confident FP vs true pathogenic vs ordinary benign. 4B set. Reads S3 + myvariant +
AlphaFold (cached). CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter26_structure.py
"""

from __future__ import annotations

import os
import re
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
AFC = Path("scratch/issue302/afcache")
N = 90
HGVS = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


def _af_struct(acc):
    f = AFC / f"{acc}.pdb"
    if not f.exists():
        try:
            api = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=30).json()
            if not api:
                f.write_text("")
                return None
            f.write_text(requests.get(api[0]["pdbUrl"], timeout=30).text)
        except Exception:
            return None
    txt = f.read_text()
    if not txt:
        return None
    return {int(ln[22:26]): (ln[17:20].strip(), float(ln[60:66])) for ln in txt.splitlines() if ln.startswith("ATOM") and ln[12:16].strip() == "CA"}


def _plddt(db):
    accs = list(dict.fromkeys(u["acc"] for u in (db.get("uniprot") or []) if isinstance(u, dict)))
    hgl = db.get("hgvsp") or []
    hgl = hgl if isinstance(hgl, list) else [hgl]
    cands = [(m.group(1).upper(), int(m.group(2))) for h in hgl if (m := HGVS.match(h or ""))]
    for acc in accs:
        st = _af_struct(acc)
        if not st:
            continue
        for aa3, pos in cands:
            if pos in st and st[pos][0] == aa3:
                return st[pos][1]
    return None


def main() -> None:
    AFC.mkdir(parents=True, exist_ok=True)
    m = pl.read_parquet(S).filter(pl.col("subset") == "missense_variant").with_columns(pl.col("chrom").cast(str), (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"))
    pmd = m.filter(pl.col("label") == 1)["mll"].median()
    sets = {
        "FP": m.filter((pl.col("label") == 0) & (pl.col("mll") >= pmd)).sort("mll", descending=True).head(N),
        "TP": m.filter((pl.col("label") == 1) & (pl.col("mll") >= pmd)).sort("mll", descending=True).head(N),
        "B0": m.filter((pl.col("label") == 0) & (pl.col("mll") < pmd)).head(N),
    }
    allrows = [{**r, "grp": g} for g, df in sets.items() for r in df.iter_rows(named=True)]
    ids = [f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}" for r in allrows]
    resp = requests.post("https://myvariant.info/v1/variant", data={"ids": ",".join(ids), "fields": "dbnsfp.uniprot,dbnsfp.hgvsp", "assembly": "hg38"}, timeout=180).json()
    byid = {d.get("query"): d for d in resp}

    vals = {g: [] for g in sets}
    for r in allrows:
        d = byid.get(f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}", {})
        pl_ = _plddt(d.get("dbnsfp", {}) or {})
        if pl_ is not None:
            vals[r["grp"]].append(pl_)
    for g in ("TP", "FP", "B0"):
        v = np.array(vals[g])
        print(f"{g}: n_mapped={len(v)}/{N}  median pLDDT={np.median(v):.1f}  %disordered(pLDDT<50)={np.mean(v < 50) * 100:.0f}%  %low(<70)={np.mean(v < 70) * 100:.0f}%")
    u_ft, p_ft = mannwhitneyu(vals["FP"], vals["TP"], alternative="two-sided")
    print(f"MWU pLDDT FP vs TP: p={p_ft:.2e}")
    pl.DataFrame([{"grp": g, "plddt": v} for g in sets for v in vals[g]]).write_parquet("scratch/issue302/structure_plddt.parquet")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    data = [vals[g] for g in ("TP", "FP", "B0")]
    parts = ax.violinplot(data, showmedians=True)
    for pc, c in zip(parts["bodies"], ["tab:red", "tab:orange", "tab:blue"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax.axhspan(0, 50, color="gray", alpha=0.12)
    ax.text(3.4, 35, "disordered\n(pLDDT<50)", fontsize=8, color="gray", ha="right")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"TP\n(med {np.median(vals['TP']):.0f}, n={len(vals['TP'])})", f"FP\n(med {np.median(vals['FP']):.0f}, n={len(vals['FP'])})", f"B0\n(med {np.median(vals['B0']):.0f}, n={len(vals['B0'])})"], fontsize=9)
    ax.set_ylabel("AlphaFold pLDDT at the residue (low = disordered/flexible)")
    ax.set_title(f"Are the over-called benigns at disordered positions?\nMWU FP vs TP p={p_ft:.1e}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "structure_plddt.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "structure_plddt.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'structure_plddt'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in ("scratch/issue302/structure_plddt.parquet", str(OUT / "structure_plddt.png"), str(OUT / "structure_plddt.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
