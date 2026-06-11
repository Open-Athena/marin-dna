"""issue #302 — iteration 28: what actually separates the model's confident FALSE POSITIVES
from true pathogenics? A head-to-head AUROC of every available signal, within the confident
set (LLR >= pathogenic median, so the gLM's own readout is ~uninformative by construction).
Synthesizes the investigation and points the fix: structure + human-population data separate
them; the gLM's likelihood/representation barely do.

Signals: AlphaFold pLDDT & burial (structure, iter26/27), AlphaMissense & REVEL (human-aware
VEPs, iter22), gLM zero-shot minus_llr (the failing readout), and the gLM frozen-embedding
probe (cited from iter11/13). 4B. Reads S3 + myvariant + cached AlphaFold. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter28_discriminators.py
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

S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
AFC = Path("scratch/issue302/afcache")
N = 110
HGVS = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
EMB_PROBE_AUROC = (
    0.66  # gLM frozen-embedding probe, within-confident TP-vs-FP (iter11/13)
)
_C: dict = {}


def _struct(acc):
    if acc in _C:
        return _C[acc]
    f = AFC / f"{acc}.pdb"
    if not f.exists():
        try:
            api = requests.get(
                f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=30
            ).json()
            f.write_text(requests.get(api[0]["pdbUrl"], timeout=30).text if api else "")
        except Exception:
            _C[acc] = None
            return None
    txt = f.read_text()
    if not txt:
        _C[acc] = None
        return None
    res, xyz = {}, []
    for ln in txt.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            res[int(ln[22:26])] = (
                ln[17:20].strip(),
                float(ln[60:66]),
                (float(ln[30:38]), float(ln[38:46]), float(ln[46:54])),
            )
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    _C[acc] = (res, np.array(xyz))
    return _C[acc]


def _structfeat(db):
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
        st = _struct(acc)
        if not st:
            continue
        res, allxyz = st
        for aa3, pos in cands:
            if pos in res and res[pos][0] == aa3:
                p = np.array(res[pos][2])
                return res[pos][1], int(
                    (np.linalg.norm(allxyz - p, axis=1) < 10).sum()
                ) - 1
    return None, None


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
    tp = (
        m.filter((pl.col("label") == 1) & (pl.col("mll") >= pmd))
        .sort("mll", descending=True)
        .head(N)
    )
    rows = [{**r, "y": 0} for r in fp.iter_rows(named=True)] + [
        {**r, "y": 1} for r in tp.iter_rows(named=True)
    ]
    ids = [f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}" for r in rows]
    byid = {
        d.get("query"): d
        for d in requests.post(
            "https://myvariant.info/v1/variant",
            data={
                "ids": ",".join(ids),
                "fields": "dbnsfp.uniprot,dbnsfp.hgvsp,dbnsfp.alphamissense.score,dbnsfp.revel.score",
                "assembly": "hg38",
            },
            timeout=180,
        ).json()
    }

    rec = []
    for r in rows:
        d = byid.get(f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}", {})
        db = d.get("dbnsfp", {}) or {}
        plddt, burial = _structfeat(db)
        rec.append(
            {
                "y": r["y"],
                "pLDDT": (plddt if plddt is not None else None),
                "burial": (float(burial) if burial is not None else None),
                "alphamissense": _fmax(db.get("alphamissense")),
                "revel": _fmax(db.get("revel")),
                "gLM_LLR": r["mll"],
            }
        )
    R = pl.DataFrame(rec)
    y = R["y"].to_numpy()
    feats = {
        "AlphaFold pLDDT\n(structure)": "pLDDT",
        "AlphaFold burial\n(buried core)": "burial",
        "AlphaMissense": "alphamissense",
        "REVEL": "revel",
        "gLM zero-shot LLR": "gLM_LLR",
    }
    aurocs = {}
    for lab, col in feats.items():
        v = R[col].to_numpy().astype(float)
        msk = ~np.isnan(v)
        aurocs[lab] = roc_auc_score(y[msk], v[msk])
        print(
            f"  {lab.splitlines()[0]:>18}: AUROC(TP vs FP) = {aurocs[lab]:.3f}  (n={msk.sum()})"
        )
    aurocs["gLM embedding probe\n(iter11/13)"] = EMB_PROBE_AUROC
    print(f"  gLM embedding probe (cited): {EMB_PROBE_AUROC}")

    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(aurocs, key=lambda k: aurocs[k])
    colors = [
        "tab:green"
        if "AlphaFold" in k or "Missense" in k or "REVEL" in k
        else "tab:red"
        for k in order
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.barh(range(len(order)), [aurocs[k] for k in order], color=colors)
    for i, k in enumerate(order):
        ax.text(aurocs[k] + 0.005, i, f"{aurocs[k]:.2f}", va="center", fontsize=9)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel(
        "AUROC separating true-pathogenic from confident-FP (within the confident set)"
    )
    ax.set_title(
        "What separates the gLM's confident false positives from true pathogenics?\nStructure + human-aware VEPs (green) do; the gLM's own readout/representation (red) barely"
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "discriminators.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "discriminators.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'discriminators'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "discriminators.png"), str(OUT / "discriminators.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
