"""Frozen-probe + zero-shot-LLR chr1-fold AUPRC over a (subset x train_frac) grid (#369).

Feeds two figures: the **data-scaling probe curve** (missense x fractions) and the
**per-subset probe baselines** (all subsets, full data) that the pooled-all FT is compared
against. CPU-only and embarrassingly parallel (GridSearchCV over the C-grid, n_jobs=-1) —
run on a many-core box, not the GPU.

  python scripts/issue369/probe_grid.py --model 255M --subsets missense_variant --train-fracs 0.125,0.25,0.5,1.0
  python scripts/issue369/probe_grid.py --model 255M --subsets all --train-fracs 1.0
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.evals.variant_probe import (
    DEFAULT_C_GRID,
    _inner_c_search,
    pair_feature_from_bundle,
)

SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
STEMS = {"46M": "h640-p46M", "76M": "h768-p76M", "128M": "h896-p128M",
         "255M": "h1152-p255M", "476M": "h1408-p476M", "1B": "h1920-p1B"}
N_JOBS = -1  # dedicated many-core CPU box


def _subsample(idx: np.ndarray, label: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(1000 + seed)
    return np.sort(np.concatenate([
        rng.choice(ci, max(1, round(len(ci) * frac)), replace=False)
        for c in (0, 1) for ci in [idx[label[idx] == c]]
    ]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="255M")
    ap.add_argument("--subsets", default="missense_variant", help="comma list or 'all'")
    ap.add_argument("--train-fracs", default="1.0")
    ap.add_argument("--pooled", action="store_true", help="ONE probe on all subsets, eval per-subset")
    ap.add_argument("--out", default="s3://oa-bolinas/analysis/issue369/probe_grid")
    args = ap.parse_args()

    path = f"{SCORES}/scaling-v0.5-{STEMS[args.model]}-step-215573/mendelian_traits.parquet"
    df = pl.read_parquet(path).to_pandas()
    subs = sorted(df["subset"].unique()) if args.subsets == "all" else args.subsets.split(",")
    fracs = [float(x) for x in args.train_fracs.split(",")]

    if args.pooled:  # ONE probe trained on all subsets pooled, evaluated per subset
        keep = [s for s in df["subset"].unique() if len(df[df["subset"] == s]) >= 100]
        pool = df[df["subset"].isin(keep)].reset_index(drop=True)
        feat = pair_feature_from_bundle(pool, "concat_ref_delta")
        label = pool["label"].to_numpy().astype(int)
        chrom = pool["chrom"].astype(str).to_numpy()
        ss = pool["subset"].astype(str).to_numpy()
        chr1 = chrom == "1"
        tr_all = np.where(~chr1)[0]
        prows = []
        for f in fracs:
            tr = _subsample(tr_all, label, f)
            gs = _inner_c_search(DEFAULT_C_GRID, 5, N_JOBS)
            gs.fit(feat[tr], label[tr], groups=chrom[tr])
            oof = gs.predict_proba(feat[chr1])[:, 1]
            lab1, ss1, ch1 = label[chr1], ss[chr1], chrom[chr1]
            for s in sorted(set(ss1.tolist())):
                m = ss1 == s
                if m.sum() >= 30 and 0 < lab1[m].sum() < m.sum():
                    ap = float(per_chrom_weighted_ap(lab1[m], oof[m], ch1[m]))
                    prows.append({"model": args.model, "subset": s, "train_frac": f,
                                  "probe_pooled_chr1": round(ap, 3), "n_train": int(len(tr))})
                    print(f"[probe-pooled] {args.model} {s} tf={f}: {ap:.3f}", flush=True)
        pl.DataFrame(prows).write_parquet(f"{args.out}/{args.model}_pooled.parquet")
        print(f"[probe] saved pooled -> {args.out}/{args.model}_pooled.parquet", flush=True)
        return

    rows = []
    for s in subs:
        sub = df[df["subset"] == s].reset_index(drop=True)
        if len(sub) < 100:
            print(f"[probe] skip {s} (n={len(sub)})", flush=True)
            continue
        label = sub["label"].to_numpy().astype(int)
        chrom = sub["chrom"].astype(str).to_numpy()
        feat = pair_feature_from_bundle(sub, "concat_ref_delta")
        minus_llr = -((sub["llr_fwd"].to_numpy() + sub["llr_rc"].to_numpy()) / 2)
        chr1 = chrom == "1"
        if not (0 < label[chr1].sum() < chr1.sum()):
            print(f"[probe] skip {s}: chr1 single-class", flush=True)
            continue
        llr = per_chrom_weighted_ap(label[chr1], minus_llr[chr1], chrom[chr1])
        tr_all = np.where(~chr1)[0]
        for f in fracs:
            tr = _subsample(tr_all, label, f)
            gs = _inner_c_search(DEFAULT_C_GRID, 5, N_JOBS)
            gs.fit(feat[tr], label[tr], groups=chrom[tr])
            oof = gs.predict_proba(feat[chr1])[:, 1]
            probe = per_chrom_weighted_ap(label[chr1], oof, chrom[chr1])
            rows.append({"model": args.model, "subset": s, "train_frac": f,
                         "n_train": int(len(tr)), "n_chr1": int(chr1.sum()),
                         "n_pos_chr1": int(label[chr1].sum()), "probe_chr1": probe,
                         "llr_chr1": llr})
            print(f"[probe] {args.model} {s} tf={f}: probe={probe:.3f} llr={llr:.3f} "
                  f"(n_train={len(tr)})", flush=True)

    dest = f"{args.out}/{args.model}_{args.subsets}.parquet"
    pl.DataFrame(rows).write_parquet(dest)
    print(f"[probe] saved -> {dest}", flush=True)


if __name__ == "__main__":
    main()
