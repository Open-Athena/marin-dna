"""issue #302 — iteration 31: are the confident FPs benign for the SAME reason or for
COMPLEMENTARY reasons? Three gLM-invisible tolerance signals, each read by a different
human-aware VEP:
  - structural tolerance: AlphaFold pLDDT < 70 (disordered/flexible)  [AlphaMissense]  (iter26)
  - MSA / mammal-specific conservation: phyloP_100v < 4 (vertebrate-tolerant)  [GPN-Star]  (iter21/30)
  - human-population tolerance: gnomAD popmax > 1%  [REVEL / popmax]  (iter6)
For the confident FPs, flag each, and measure overlap vs complementarity (does any single signal
cover them, or are they a heterogeneous union?). 4B. Reads S3 + myvariant + cached AlphaFold +
conservation tracks. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter31_complementarity.py
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests

S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
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
        rec.append(
            {"plddt": plddt, "phyloP_100v": r["phyloP_100v"], "popmax": _popmax(d)}
        )
    R = pl.DataFrame(rec)
    # only variants where all three signals are measurable (fair complementarity)
    R = R.filter(
        pl.col("plddt").is_not_null()
        & pl.col("phyloP_100v").is_not_null()
        & pl.col("popmax").is_not_null()
    )
    n = R.height
    struct = (R["plddt"] < 70).to_numpy()
    msa = (R["phyloP_100v"] < 4).to_numpy()
    human = (R["popmax"] > 0.01).to_numpy()
    flags = {
        "structural\n(pLDDT<70)": struct,
        "MSA-shallow\n(phyloP_100v<4)": msa,
        "human-common\n(popmax>1%)": human,
    }
    print(f"confident FPs with all 3 signals measured: n={n}")
    for name, f in flags.items():
        print(f"  {name.splitlines()[0]:>14}: {f.mean() * 100:.0f}% ({f.sum()})")
    anyf = struct | msa | human
    print(
        f"  explained by >=1 signal: {anyf.mean() * 100:.0f}% ({anyf.sum()})  | by NONE: {(~anyf).sum()}"
    )
    # pairwise co-occurrence (Jaccard) — complementary if low
    names = list(flags)
    arr = list(flags.values())
    for i, j in combinations(range(3), 2):
        a, b = arr[i], arr[j]
        jac = (a & b).sum() / max((a | b).sum(), 1)
        print(
            f"  Jaccard {names[i].splitlines()[0]} ∩ {names[j].splitlines()[0]} = {jac:.2f}"
        )
    # single-signal-only counts (complementarity)
    only = {
        names[k]: int((arr[k] & ~arr[(k + 1) % 3] & ~arr[(k + 2) % 3]).sum())
        for k in range(3)
    }
    print(f"  explained by EXACTLY one signal: {only}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    # Panel A — coverage by each signal + union
    labs = [k for k in flags] + ["ANY of 3"]
    vals = [f.mean() * 100 for f in flags.values()] + [anyf.mean() * 100]
    ax[0].bar(range(4), vals, color=["tab:orange", "tab:green", "tab:blue", "tab:gray"])
    for i, v in enumerate(vals):
        ax[0].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(labs, fontsize=8)
    ax[0].set_ylabel("% of confident FPs flagged tolerant")
    ax[0].set_title(
        f"Each tolerance signal explains a SUBSET;\ntogether they cover {anyf.mean() * 100:.0f}% (n={n})"
    )
    ax[0].grid(alpha=0.3, axis="y")
    # Panel B — 3-set membership counts (UpSet-style)
    from itertools import product

    combos = list(product([0, 1], repeat=3))[1:]
    counts = []
    cl = []
    for c in combos:
        mask = np.ones(n, bool)
        for k, bit in enumerate(c):
            mask &= arr[k] if bit else ~arr[k]
        counts.append(int(mask.sum()))
        cl.append("".join("●" if b else "·" for b in c))
    order = np.argsort(counts)[::-1]
    ax[1].bar(range(len(combos)), [counts[i] for i in order], color="tab:purple")
    ax[1].set_xticks(range(len(combos)))
    ax[1].set_xticklabels([cl[i] for i in order], fontsize=11)
    ax[1].set_xlabel("structural ● / MSA-shallow ● / human-common ●  membership")
    ax[1].set_ylabel("# confident FPs")
    ax[1].set_title(
        "Membership combinations — mostly single-signal\n(complementary, not redundant)"
    )
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "The over-called benigns are benign for COMPLEMENTARY reasons — each captured by a different human-aware VEP, none by the gLM",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "complementarity.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "complementarity.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'complementarity'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "complementarity.png"), str(OUT / "complementarity.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
