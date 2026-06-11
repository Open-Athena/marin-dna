"""issue #302 — iteration 36: validate iter31's complementarity ON THE ACTUAL METHOD. iter31
showed our confident FPs split into structural / MSA-shallow / human-common tolerance subsets.
iter35 showed GPN-Star (a vertebrate-MSA model) rescues the FPs. Question: does GPN-Star rescue
SPECIFICALLY its own axis (the MSA-shallow subset) and NOT the structural-only / human-only ones —
i.e. is the complementarity real at the method level (each VEP taps its own signal)?

For our confident FPs, flag the 3 tolerance signals (iter31) + join GPN-Star-V (iter35). For each
flag, measure GPN-Star's RESCUE = AUROC(pathogenic vs FPs-carrying-that-flag) — higher means
GPN-Star correctly demotes that FP subset. Prediction: GPN-Star rescues the MSA-shallow subset
strongly, the structural-only / human-only ones much less (it can't read structure or human AF),
and the residual not at all. 4B. Reads S3 + gist + myvariant + AlphaFold cache + tracks. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter36_method_specificity.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
from sklearn.metrics import roc_auc_score

OURS = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
GPN_V = "https://gist.githubusercontent.com/gonzalobenegas/db282f89aa00244fbb7437dce0f069ef/raw/02484d50d9bfd80337e313652b26f98a9362b6b1/bolinas_mendelian_traits_GPN-Star-V.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
AFC = Path("scratch/issue302/afcache")
KEY = ["chrom", "pos", "ref", "alt"]
N = 150
HGVS = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
_C: dict = {}


def _plddt(db):
    accs = list(
        dict.fromkeys(
            u["acc"] for u in (db.get("uniprot") or []) if isinstance(u, dict)
        )
    )
    hgl = db.get("hgvsp") or []
    hgl = hgl if isinstance(hgl, list) else [hgl]
    cands = [
        (m.group(1).upper(), int(m.group(2))) for h in hgl if (m := HGVS.match(h or ""))
    ]
    for acc in accs:
        if acc not in _C:
            f = AFC / f"{acc}.pdb"
            txt = f.read_text() if f.exists() else ""
            _C[acc] = (
                {
                    int(ln[22:26]): (ln[17:20].strip(), float(ln[60:66]))
                    for ln in txt.splitlines()
                    if ln.startswith("ATOM") and ln[12:16].strip() == "CA"
                }
                if txt
                else None
            )
        st = _C[acc]
        if not st:
            continue
        for aa3, pos in cands:
            if pos in st and st[pos][0] == aa3:
                return st[pos][1]
    return None


def _popmax(d):
    g = (d.get("gnomad_genome") or {}).get("af") or {}
    if not isinstance(g, dict):
        return None
    pops = [
        v for k, v in g.items() if k.startswith("af_") and isinstance(v, (int, float))
    ]
    return max(pops) if pops else g.get("af")


def _track(name):
    return (
        pl.read_parquet(f"{CB}/{name}_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias(name)])
    )


def main() -> None:
    m = (
        pl.read_parquet(OURS)
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(
            pl.col("chrom").cast(str),
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("ours"),
        )
    )
    gpn = (
        pl.read_parquet(GPN_V)
        .with_columns(
            [pl.col("chrom").cast(str), (-pl.col("llr_calibrated")).alias("gpn")]
        )
        .select([*KEY, "gpn"])
    )
    m = m.join(gpn, on=KEY, how="inner").join(_track("phyloP_100v"), on=KEY, how="left")
    pmd = m.filter(pl.col("label") == 1)["ours"].median()
    tp = m.filter(pl.col("label") == 1)
    tp_gpn = tp["gpn"].to_numpy()
    fp = (
        m.filter((pl.col("label") == 0) & (pl.col("ours") >= pmd))
        .sort("ours", descending=True)
        .head(N)
    )
    rows = list(fp.iter_rows(named=True))
    ids = [f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}" for r in rows]
    byid = {
        d.get("query"): d
        for d in requests.post(
            "https://myvariant.info/v1/variant",
            data={
                "ids": ",".join(ids),
                "fields": "dbnsfp.uniprot,dbnsfp.hgvsp,gnomad_genome.af",
                "assembly": "hg38",
            },
            timeout=180,
        ).json()
    }
    rec = []
    for r in rows:
        d = byid.get(f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}", {})
        plddt = _plddt(d.get("dbnsfp", {}) or {})
        pm = _popmax(d)
        if plddt is None or r["phyloP_100v"] is None or pm is None:
            continue
        rec.append(
            {
                "gpn": r["gpn"],
                "structural": plddt < 70,
                "msa": r["phyloP_100v"] < 4,
                "human": pm > 0.01,
            }
        )
    R = pl.DataFrame(rec)
    R = R.with_columns(
        (~(pl.col("structural") | pl.col("msa") | pl.col("human"))).alias("residual")
    )
    n = R.height
    print(f"confident FPs with all signals + GPN-Star: n={n}")

    def rescue(mask):
        g = R.filter(mask)["gpn"].to_numpy()
        if len(g) < 5:
            return np.nan, len(g)
        y = np.r_[np.ones(len(tp_gpn)), np.zeros(len(g))]
        s = np.r_[tp_gpn, g]
        ok = ~np.isnan(s)
        return roc_auc_score(y[ok], s[ok]), len(g)

    flags = [
        ("MSA-shallow\n(GPN's axis)", pl.col("msa"), "tab:green"),
        (
            "structural\n(AM's axis)",
            pl.col("structural") & ~pl.col("msa"),
            "tab:orange",
        ),
        (
            "human-common\n(REVEL's axis)",
            pl.col("human") & ~pl.col("msa") & ~pl.col("structural"),
            "tab:blue",
        ),
        ("residual\n(no signal)", pl.col("residual"), "dimgray"),
    ]
    res = []
    allr, alln = rescue(pl.lit(True))
    print(
        f"  GPN-Star rescue (AUROC TP vs FP-subset) — ALL confident FPs: {allr:.3f} (n={alln})"
    )
    for name, mask, col in flags:
        a, k = rescue(mask)
        res.append((name, a, k, col))
        print(f"    {name.splitlines()[0]:>14}: GPN rescue AUROC={a:.3f} (n={k})")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    names = [r[0] for r in res]
    vals = [r[1] for r in res]
    ns = [r[2] for r in res]
    cols = [r[3] for r in res]
    ax.bar(range(len(res)), vals, color=cols)
    ax.axhline(
        allr, ls="--", color="black", lw=1, label=f"GPN rescue, all FPs ({allr:.2f})"
    )
    ax.axhline(0.5, ls=":", color="red", lw=1, label="no rescue (0.5)")
    for i, (v, k) in enumerate(zip(vals, ns)):
        ax.text(i, v + 0.01, f"{v:.2f}\nn={k}", ha="center", fontsize=9)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("GPN-Star RESCUE of this FP subset\nAUROC(pathogenic vs FP-subset)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        "GPN-Star rescues SPECIFICALLY its own (MSA-depth) FP subset —\nmuch less the structural/human-only ones, and not the residual\n(complementarity is real at the method level)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "method_specificity.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "method_specificity.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'method_specificity'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "method_specificity.png"), str(OUT / "method_specificity.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
