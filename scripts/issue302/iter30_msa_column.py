"""issue #302 — iteration 30: the whole-genome ALIGNMENT COLUMN at the FP sites (GB's steer).
GPN-Star catches these FPs WITHOUT structure — it uses a vertebrate MSA — so the alignment
column must carry a signal our single-sequence gLM lacks. Three questions, all answered by
the column:
  1. Is the ALT allele actually PRESENT in other species? (the "model should know it's viable"
     signal an MSA model has and a single-sequence gLM doesn't)
  2. What do human-aware/MSA methods hold onto? -> exactly this column pattern.
  3. Are the "conserved" FPs conserved by chance/drift? -> alignment depth + column diversity.

Per variant: query the Ensembl EPO mammals alignment (69-way, JSON) at the exact position,
read the homologous base in each extant species. Compare confident FP vs true pathogenic (TP)
vs ordinary benign (B0). 4B. Reads S3 + Ensembl REST (cached). CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter30_msa_column.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
from scipy.stats import fisher_exact, mannwhitneyu

S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
EPO = Path("scratch/issue302/epocache")
N = 90


def _column(chrom, pos):
    """Extant-species bases (uppercase) at chrom:pos (1-based) from the EPO mammals alignment.
    Ensembl needs a >=3bp region, so query [pos-1, pos+1] and take the CENTER column (the 2nd
    non-gap human base), which handles insertions/gaps in other species."""
    f = EPO / f"{chrom}_{pos}.json"
    if f.exists():
        d = json.loads(f.read_text())
    else:
        try:
            r = requests.get(
                f"https://rest.ensembl.org/alignment/region/homo_sapiens/{chrom}:{pos - 1}-{pos + 1}",
                params={"method": "EPO", "species_set_group": "mammals"},
                headers={"Content-Type": "application/json"},
                timeout=40,
            )
            d = r.json() if r.status_code == 200 else []
        except Exception:
            d = []
        f.write_text(json.dumps(d))
        time.sleep(0.06)
    if not d:
        return None, None
    al = d[0].get("alignments", [])
    human = next((a for a in al if a.get("species") == "homo_sapiens"), None)
    if not human:
        return None, None
    hseq = (human.get("seq") or "").upper()
    nongap = [i for i, c in enumerate(hseq) if c != "-"]
    if len(nongap) < 2:
        return None, None
    col = nongap[1]  # center of the 3bp window = the variant position
    bases = {}
    for a in al:
        sp = a.get("species", "")
        if (
            "[" in sp
            or sp == "homo_sapiens"
            or str(a.get("seq_region", "")).startswith("Ancestor")
        ):
            continue
        sq = (a.get("seq") or "").upper()
        bases[sp] = sq[col] if col < len(sq) else "-"
    return bases, hseq[col]


def main() -> None:
    m = (
        pl.read_parquet(S)
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(
            pl.col("chrom").cast(str),
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
        )
    )
    pmd = m.filter(pl.col("label") == 1)["mll"].median()
    sets = {
        "FP": m.filter((pl.col("label") == 0) & (pl.col("mll") >= pmd))
        .sort("mll", descending=True)
        .head(N),
        "TP": m.filter((pl.col("label") == 1) & (pl.col("mll") >= pmd))
        .sort("mll", descending=True)
        .head(N),
        "B0": m.filter((pl.col("label") == 0) & (pl.col("mll") < pmd)).head(N),
    }
    rec = {g: [] for g in sets}
    for g, df in sets.items():
        for r in df.iter_rows(named=True):
            bases, href = _column(r["chrom"], r["pos"])
            if not bases:
                continue
            vals = [b for b in bases.values() if b in "ACGT"]
            if len(vals) < 5:
                continue
            ref, alt = r["ref"].upper(), r["alt"].upper()
            rec[g].append(
                {
                    "n_sp": len(vals),
                    "ref_frac": np.mean([b == ref for b in vals]),
                    "alt_present": int(any(b == alt for b in vals)),
                    "n_distinct": len(set(vals)),
                    "href_ok": href == ref,
                }
            )
        a = rec[g]
        print(
            f"{g}: n={len(a)} | alt_present={np.mean([x['alt_present'] for x in a]) * 100:.0f}%  median ref_frac={np.median([x['ref_frac'] for x in a]):.2f}  median n_sp={np.median([x['n_sp'] for x in a]):.0f}  median distinct={np.median([x['n_distinct'] for x in a]):.1f}  href_ok={np.mean([x['href_ok'] for x in a]) * 100:.0f}%"
        )

    def col(g, k):
        return np.array([x[k] for x in rec[g]])

    a_fp, a_tp = col("FP", "alt_present"), col("TP", "alt_present")
    orr, pf = fisher_exact(
        [[a_fp.sum(), len(a_fp) - a_fp.sum()], [a_tp.sum(), len(a_tp) - a_tp.sum()]]
    )
    print(
        f"\nFisher alt_present FP vs TP: OR={orr:.2f} p={pf:.2e}  (FP {a_fp.mean() * 100:.0f}% vs TP {a_tp.mean() * 100:.0f}%)"
    )
    p_rf = mannwhitneyu(col("FP", "ref_frac"), col("TP", "ref_frac"))[1]
    print(f"MWU ref_frac FP vs TP: p={p_rf:.2e}")
    pl.DataFrame([{"grp": g, **x} for g in sets for x in rec[g]]).write_parquet(
        "scratch/issue302/msa_column.parquet"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    grps = ["TP", "FP", "B0"]
    cols = ["tab:red", "tab:orange", "tab:blue"]
    ax[0].bar(range(3), [col(g, "alt_present").mean() * 100 for g in grps], color=cols)
    for i, g in enumerate(grps):
        ax[0].text(
            i,
            col(g, "alt_present").mean() * 100 + 1,
            f"{col(g, 'alt_present').mean() * 100:.0f}%",
            ha="center",
            fontsize=10,
        )
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([f"{g}\n(n={len(rec[g])})" for g in grps])
    ax[0].set_ylabel("% of variants where the ALT base is present in ≥1 other mammal")
    ax[0].set_title(
        f"Is the deleterious-looking ALT actually seen in other species?\nFP vs TP Fisher p={pf:.1e}  (an MSA model knows; a single-seq gLM doesn't)"
    )
    ax[0].grid(alpha=0.3, axis="y")
    parts = ax[1].violinplot([col(g, "ref_frac") for g in grps], showmedians=True)
    for pc, c in zip(parts["bodies"], cols):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax[1].set_xticks(range(1, 4))
    ax[1].set_xticklabels(
        [f"{g}\n(med {np.median(col(g, 'ref_frac')):.2f})" for g in grps]
    )
    ax[1].set_ylabel("fraction of mammals sharing the human REF (column invariance)")
    ax[1].set_title("How 'real'/deep is the conservation at these sites?")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "The alignment column at the FP sites — the MSA signal a single-sequence gLM lacks",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "msa_column.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "msa_column.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'msa_column'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/msa_column.parquet",
        str(OUT / "msa_column.png"),
        str(OUT / "msa_column.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
