"""issue #302 — iteration 27: sharpen iter26 — the over-called benigns are at disordered OR
SURFACE positions; true pathogenics are at ordered+buried cores. iter26 showed FPs are more
disordered (low pLDDT), but ~half are at ORDERED positions (pLDDT>70) yet still tolerated.
This adds BURIAL (CA contact number from the AlphaFold structure; high = buried core, low =
surface) to test whether the ordered FPs are at the tolerant SURFACE.

Reuses the cached AlphaFold structures (iter26). Per variant: validated residue -> (pLDDT,
burial). 2D structural map for FP / TP / B0. 4B confident set. Reads S3 + myvariant + cached
AlphaFold PDBs. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter27_burial.py
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
from scipy.stats import mannwhitneyu

S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
AFC = Path("scratch/issue302/afcache")
N = 90
HGVS = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
_CACHE: dict = {}


def _struct(acc):
    if acc in _CACHE:
        return _CACHE[acc]
    f = AFC / f"{acc}.pdb"
    if not f.exists():
        try:
            api = requests.get(
                f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=30
            ).json()
            f.write_text(requests.get(api[0]["pdbUrl"], timeout=30).text if api else "")
        except Exception:
            _CACHE[acc] = None
            return None
    txt = f.read_text()
    if not txt:
        _CACHE[acc] = None
        return None
    res, coords = {}, []
    for ln in txt.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            rs = int(ln[22:26])
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            res[rs] = (ln[17:20].strip(), float(ln[60:66]), xyz)
            coords.append((rs, xyz))
    _CACHE[acc] = (res, np.array([c[1] for c in coords]), [c[0] for c in coords])
    return _CACHE[acc]


def _lookup(db):
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
        res, allxyz, _ = st
        for aa3, pos in cands:
            if pos in res and res[pos][0] == aa3:
                plddt, xyz = res[pos][1], np.array(res[pos][2])
                contact = (
                    int((np.linalg.norm(allxyz - xyz, axis=1) < 10.0).sum()) - 1
                )  # CA neighbors within 10A
                return plddt, contact
    return None


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
    allrows = [
        {**r, "grp": g} for g, df in sets.items() for r in df.iter_rows(named=True)
    ]
    ids = [f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}" for r in allrows]
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

    pts = {g: [] for g in sets}
    for r in allrows:
        d = byid.get(f"chr{r['chrom']}:g.{r['pos']}{r['ref']}>{r['alt']}", {})
        hit = _lookup(d.get("dbnsfp", {}) or {})
        if hit:
            pts[r["grp"]].append(hit)
    for g in ("TP", "FP", "B0"):
        a = np.array(pts[g])
        print(
            f"{g}: n={len(a)}  pLDDT med={np.median(a[:, 0]):.0f}  contact med={np.median(a[:, 1]):.0f}  | ordered(pLDDT>70): n={int((a[:, 0] > 70).sum())} contact med={np.median(a[a[:, 0] > 70][:, 1]):.0f}"
        )
    # among ORDERED residues (pLDDT>70), is FP burial < TP burial (surface)?
    fo = np.array(pts["FP"])
    to = np.array(pts["TP"])
    fo_ord, to_ord = fo[fo[:, 0] > 70][:, 1], to[to[:, 0] > 70][:, 1]
    p_ord = mannwhitneyu(fo_ord, to_ord, alternative="less")[1]
    print(
        f"\nordered-only (pLDDT>70) burial: FP med={np.median(fo_ord):.0f} (n={len(fo_ord)}) vs TP med={np.median(to_ord):.0f} (n={len(to_ord)})  MWU(FP<TP) p={p_ord:.2e}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
    col = {"TP": "tab:red", "FP": "tab:orange", "B0": "tab:blue"}
    for g in ("B0", "FP", "TP"):
        a = np.array(pts[g])
        ax[0].scatter(
            a[:, 0],
            a[:, 1],
            s=28,
            alpha=0.6,
            color=col[g],
            label=f"{g} (n={len(a)})",
            edgecolors="none",
        )
    ax[0].axvline(70, color="gray", ls=":", lw=1)
    ax[0].axhline(np.median(np.array(pts["TP"])[:, 1]), color="gray", ls=":", lw=1)
    ax[0].set_xlabel("AlphaFold pLDDT (low = disordered →)")
    ax[0].set_ylabel("burial (CA contacts <10Å; high = buried core)")
    ax[0].set_title(
        "Structural map: pathogenics in the ordered+buried core;\nFPs escape to disordered (left) OR surface (bottom)"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    parts = ax[1].violinplot([fo_ord, to_ord], showmedians=True)
    for pc, c in zip(parts["bodies"], ["tab:orange", "tab:red"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax[1].set_xticks([1, 2])
    ax[1].set_xticklabels(
        [
            f"FP ordered\n(med {np.median(fo_ord):.0f}, n={len(fo_ord)})",
            f"TP ordered\n(med {np.median(to_ord):.0f}, n={len(to_ord)})",
        ],
        fontsize=9,
    )
    ax[1].set_ylabel("burial (CA contacts <10Å)")
    ax[1].set_title(
        f"Even among ORDERED positions (pLDDT>70),\nthe FPs are less buried (more surface)  MWU p={p_ord:.1e}"
    )
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "The over-called benigns are at disordered OR surface positions — true pathogenics are at ordered+buried cores",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "structure_burial.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "structure_burial.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'structure_burial'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "structure_burial.png"), str(OUT / "structure_burial.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
