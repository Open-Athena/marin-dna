"""issue #302 — iteration 32: what is iter31's ~19% RESIDUAL (confident FPs flagged by none of
structure / MSA-depth / human-pop)? Either label noise (mislabeled pathogenics → the
degradation is partly inflated) or a 4th tolerance signal. Test: do peer VEPs (AlphaMissense,
REVEL) call the residual pathogenic too, and what is its ClinVar review status?

For the confident FPs: the three tolerance flags (iter31) + AlphaMissense + REVEL + ClinVar.
Compare 'explained' (>=1 flag) vs 'residual' (none). 4B. Reads S3 + myvariant + cached
AlphaFold + conservation tracks. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter32_residual.py
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import numpy as np
import polars as pl
import requests

S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
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


def _popmax(d):
    g = (d.get("gnomad_genome") or {}).get("af") or {}
    if not isinstance(g, dict):
        return None
    pops = [
        v for k, v in g.items() if k.startswith("af_") and isinstance(v, (int, float))
    ]
    return max(pops) if pops else g.get("af")


def _clnrev(d):
    rcv = (d.get("clinvar") or {}).get("rcv")
    if not rcv:
        return None, None
    rcv = rcv if isinstance(rcv, list) else [rcv]
    sigs = [str(r.get("clinical_significance", "")) for r in rcv]
    revs = [str(r.get("review_status", "")) for r in rcv]
    return ";".join(sorted(set(s for s in sigs if s)))[:40], (revs[0] if revs else None)


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
    fp = (
        m.filter((pl.col("label") == 0) & (pl.col("mll") >= pmd))
        .sort("mll", descending=True)
        .head(N)
    )
    p100 = (
        pl.read_parquet(f"{CB}/phyloP_100v_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias("phyloP_100v")])
    )
    fp = fp.join(p100, on=KEY, how="left")
    rows = list(fp.iter_rows(named=True))
    ids = [f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}" for r in rows]
    byid = {
        d.get("query"): d
        for d in requests.post(
            "https://myvariant.info/v1/variant",
            data={
                "ids": ",".join(ids),
                "fields": "dbnsfp.uniprot,dbnsfp.hgvsp,dbnsfp.alphamissense.pred,dbnsfp.revel.score,dbnsfp.genename,gnomad_genome.af,clinvar.rcv.clinical_significance,clinvar.rcv.review_status",
                "assembly": "hg38",
            },
            timeout=180,
        ).json()
    }

    rec = []
    for r in rows:
        d = byid.get(f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}", {})
        db = d.get("dbnsfp", {}) or {}
        plddt = _plddt(db)
        pm = _popmax(d)
        if plddt is None or r["phyloP_100v"] is None or pm is None:
            continue
        amp = ((db.get("alphamissense") or {}) or {}).get("pred")
        amp = (
            collections.Counter(amp if isinstance(amp, list) else [amp]).most_common(1)[
                0
            ][0]
            if amp
            else None
        )
        cln, rev = _clnrev(d)
        gn = db.get("genename")
        rec.append(
            {
                "struct": plddt < 70,
                "msa": r["phyloP_100v"] < 4,
                "human": pm > 0.01,
                "AM": amp,
                "revel": _fmax(db.get("revel")),
                "clnsig": cln,
                "review": rev,
                "gene": (gn[0] if isinstance(gn, list) else gn),
                "id": f"{r['chrom']}:{r['pos']}{r['ref']}>{r['alt']}",
                "popmax": pm,
            }
        )
    R = pl.DataFrame(rec)
    R = R.with_columns(
        (pl.col("struct") | pl.col("msa") | pl.col("human")).alias("explained")
    )
    res = R.filter(~pl.col("explained"))
    exp = R.filter(pl.col("explained"))
    print(f"n={R.height} | explained={exp.height} residual={res.height}")
    for name, sub in (("EXPLAINED", exp), ("RESIDUAL", res)):
        amp = sub["AM"].drop_nulls().to_list()
        rev = sub["revel"].drop_nulls().to_numpy()
        print(
            f"\n{name} (n={sub.height}): AlphaMissense pathogenic={sum(a == 'P' for a in amp)}/{len(amp)} ({sum(a == 'P' for a in amp) / max(len(amp), 1) * 100:.0f}%)  REVEL>0.5={np.mean(rev > 0.5) * 100:.0f}%  median REVEL={np.median(rev):.2f}"
        )
        print(
            f"   ClinVar review status counts: {collections.Counter(sub['review'].to_list())}"
        )
    print("\nRESIDUAL variants (look pathogenic on every measured axis):")
    for r in res.iter_rows(named=True):
        print(
            f"   {r['id']:>22} {str(r['gene']):>10} AM={r['AM']} REVEL={r['revel']} popmax={r['popmax']:.4f} clinvar={r['clnsig']} review={r['review']}"
        )
    R.write_parquet("scratch/issue302/residual.parquet")


if __name__ == "__main__":
    main()
