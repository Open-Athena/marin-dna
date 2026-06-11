"""issue #302 — iteration 29: tie the structural finding (iter26/27) to the SCALE degradation
(the core #302 phenomenon). The model is sequence-only, so it should be STRUCTURE-BLIND: it
over-calls conserved benigns at structurally-tolerant (disordered/surface) positions
increasingly with scale, regardless of the structure that makes them tolerant.

For the 4B-confident FPs with an AlphaFold-mapped residue, split by structure (disordered
pLDDT<70 vs ordered >=70), and track their per-model z-standardized minus_llr (over-call
magnitude; within-model z removes score inflation, as iter8) across the ladder. If both
subsets climb similarly, the model is structure-blind — the degradation recruits
structurally-tolerant conserved positions it cannot recognize as tolerant. TPs as reference.
Reads S3 + myvariant + cached AlphaFold. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter29_structure_scale.py
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

SC = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
AFC = Path("scratch/issue302/afcache")
KEY = ["chrom", "pos", "ref", "alt"]
N = 200
HGVS = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
LADDER = [
    ("scaling-v0.5-h640-p46M-step-215573", 46),
    ("scaling-v0.5-h896-p128M-step-215573", 128),
    ("scaling-v0.5-h1152-p255M-step-215573", 255),
    ("scaling-v0.5-h1408-p476M-step-215573", 476),
    ("scaling-v0.5-h1920-p1B-step-215573", 1120),
    ("scaling-v0.5-h2432-p2B-step-215573", 2270),
    ("scaling-v0.5-h2944-p4B-step-215573", 4020),
]
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
            if not f.exists():
                try:
                    api = requests.get(
                        f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=30
                    ).json()
                    f.write_text(
                        requests.get(api[0]["pdbUrl"], timeout=30).text if api else ""
                    )
                except Exception:
                    f.write_text("")
            txt = f.read_text()
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


def main() -> None:
    base = (
        pl.read_parquet(
            f"{SC}/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
        )
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(
            pl.col("chrom").cast(str),
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
        )
    )
    pmd = base.filter(pl.col("label") == 1)["mll"].median()
    fp = (
        base.filter((pl.col("label") == 0) & (pl.col("mll") >= pmd))
        .sort("mll", descending=True)
        .head(N)
    )
    tp = (
        base.filter((pl.col("label") == 1) & (pl.col("mll") >= pmd))
        .sort("mll", descending=True)
        .head(N)
    )
    rows = [{**r, "grp": "FP"} for r in fp.iter_rows(named=True)] + [
        {**r, "grp": "TP"} for r in tp.iter_rows(named=True)
    ]
    ids = [f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}" for r in rows]
    byid = {
        d.get("query"): d
        for d in requests.post(
            "https://myvariant.info/v1/variant",
            data={
                "ids": ",".join(ids),
                "fields": "dbnsfp.uniprot,dbnsfp.hgvsp",
                "assembly": "hg38",
            },
            timeout=180,
        ).json()
    }
    for r in rows:
        r["plddt"] = _plddt(
            (
                byid.get(f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}", {}) or {}
            ).get("dbnsfp", {})
            or {}
        )
    keyset = {
        "TP": [
            (r["chrom"], r["pos"], r["ref"], r["alt"])
            for r in rows
            if r["grp"] == "TP" and r["plddt"] is not None
        ],
        "FP disordered (pLDDT<70)": [
            (r["chrom"], r["pos"], r["ref"], r["alt"])
            for r in rows
            if r["grp"] == "FP" and r["plddt"] is not None and r["plddt"] < 70
        ],
        "FP ordered (pLDDT>=70)": [
            (r["chrom"], r["pos"], r["ref"], r["alt"])
            for r in rows
            if r["grp"] == "FP" and r["plddt"] is not None and r["plddt"] >= 70
        ],
    }
    print({k: len(v) for k, v in keyset.items()})

    series = {k: [] for k in keyset}
    for sdir, params in LADDER:
        m = (
            pl.read_parquet(f"{SC}/{sdir}/mendelian_traits.parquet")
            .filter(pl.col("subset") == "missense_variant")
            .with_columns(
                pl.col("chrom").cast(str),
                (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
            )
        )
        mu, sd = m["mll"].mean(), m["mll"].std()
        zmap = {
            (r["chrom"], r["pos"], r["ref"], r["alt"]): (r["mll"] - mu) / sd
            for r in m.select([*KEY, "mll"]).iter_rows(named=True)
        }
        for k, keys in keyset.items():
            series[k].append(
                (params, float(np.mean([zmap[kk] for kk in keys if kk in zmap])))
            )

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    style = {
        "TP": ("tab:red", "D-"),
        "FP disordered (pLDDT<70)": ("tab:orange", "o-"),
        "FP ordered (pLDDT>=70)": ("tab:olive", "s-"),
    }
    for k, pts in series.items():
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(
            x,
            y,
            style[k][1],
            color=style[k][0],
            lw=2.3,
            label=f"{k} (n={len(keyset[k])})",
        )
        print(f"{k}: " + " ".join(f"{p[0]}M={p[1]:+.2f}" for p in pts))
    ax.set_xscale("log")
    ax.set_xlabel("params (M, log)")
    ax.set_ylabel("mean z-standardized minus_llr (over-call magnitude)")
    ax.set_title(
        "Structure-blind: the model over-calls conserved benigns at BOTH\ndisordered AND ordered/surface positions, increasingly with scale"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "structure_scale.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "structure_scale.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'structure_scale'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "structure_scale.png"), str(OUT / "structure_scale.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
