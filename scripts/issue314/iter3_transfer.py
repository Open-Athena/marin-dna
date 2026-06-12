"""issue #314 iter3 — variant-type transfer (one model).

Does a probe trained on one consequence transfer to others? Does a single pooled-all probe
match per-subset probes? Cross-subset, chromosome-grouped and leak-proof: to score eval-subset
B, each chromosome c of B is predicted by a probe trained on **train-subset A's variants in the
other chromosomes** (so train/eval variants never share a chromosome). OOF over B →
per-chromosome-weighted AUPRC. Fixed rep ``entire_window/abs_delta`` (symmetric → valid across
all subsets, fair for transfer); fixed strong C (the transfer *signal* is the question, not C).

Output rows = train-subset A (+ ``pooled_all`` = trained on every subset), cols = eval-subset
B; the diagonal is within-subset, off-diagonal is transfer; the LLR row is the zero-shot
baseline per eval subset. Saves a long-form parquet.
"""

import argparse

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from iter1_representation_sweep import build_feature, load_and_pool
from marin_dna.pipelines.evals.variant_probe import make_linear_probe

REP = ("pool", "entire_window", "abs_delta")
C_REG = 1e-3  # strong, from the iter1/nested good region
MIN_VARIANTS = 100


def per_chrom_weighted_ap(y: np.ndarray, score: np.ndarray, chrom: np.ndarray) -> float:
    tot, w = 0.0, 0
    for c in np.unique(chrom):
        m = chrom == c
        if 0 < int(y[m].sum()) < int(m.sum()) and np.isfinite(score[m]).all():
            tot += average_precision_score(y[m], score[m]) * int(m.sum())
            w += int(m.sum())
    return tot / w if w else float("nan")


def cross_oof(fA, yA, cA, fB, yB, cB, c_reg):
    """OOF over B: each chromosome of B is predicted by a probe trained on A's *other*
    chromosomes (leak-proof for both A==B and A!=B)."""
    oof = np.full(len(yB), np.nan)
    for c in np.unique(cB):
        tr = cA != c
        if not (0 < int(yA[tr].sum()) < int(tr.sum())):
            continue  # A lacks a class off chromosome c — can't train this fold
        probe = make_linear_probe(loss="logistic", c=c_reg, n_pca=None, standardize=True)
        probe.fit(fA[tr], yA[tr])
        m = cB == c
        oof[m] = probe.predict_proba(fB[m])[:, 1]
    return oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    pooled, inner, cov, keys = load_and_pool(args.cache)
    kdf = keys.to_pandas()
    kdf["chrom"] = kdf["chrom"].astype(str)
    feat = build_feature(pooled, inner, cov, REP)
    del pooled, inner, cov
    y_all = kdf["label"].to_numpy().astype(int)
    chrom_all = kdf["chrom"].to_numpy()
    subset_all = kdf["subset"].to_numpy()
    llr_all = (-(kdf["llr_fwd"] + kdf["llr_rc"]) / 2).to_numpy()
    subsets = [s for s in sorted(set(subset_all))
               if (subset_all == s).sum() >= MIN_VARIANTS
               and len(set(chrom_all[subset_all == s])) >= 3]
    masks = {s: (subset_all == s) for s in subsets}
    print(f"{args.model}: {len(subsets)} subsets — transfer matrix ({REP[1]}/{REP[2]}, C={C_REG})")

    rows = []
    # zero-shot LLR baseline per eval subset (per-chrom-weighted)
    for b in subsets:
        mb = masks[b]
        rows.append({"model": args.model, "train": "LLR", "eval": b,
                     "auprc": per_chrom_weighted_ap(y_all[mb], llr_all[mb], chrom_all[mb])})
    # per-subset (incl. transfer) + pooled-all train arms
    train_arms = {s: masks[s] for s in subsets}
    train_arms["pooled_all"] = np.ones(len(y_all), dtype=bool)
    for a, ma in train_arms.items():
        fA, yA, cA = feat[ma], y_all[ma], chrom_all[ma]
        line = []
        for b in subsets:
            mb = masks[b]
            oof = cross_oof(fA, yA, cA, feat[mb], y_all[mb], chrom_all[mb], C_REG)
            ap_b = per_chrom_weighted_ap(y_all[mb], oof, chrom_all[mb])
            rows.append({"model": args.model, "train": a, "eval": b, "auprc": ap_b})
            line.append(f"{b[:8]}:{ap_b:.3f}")
        print(f"  train={a:12s} " + "  ".join(line))

    pl.DataFrame(rows).write_parquet(f"{args.out}/transfer_{args.model}.parquet")
    print(f"\nwrote {args.out}/transfer_{args.model}.parquet")


if __name__ == "__main__":
    main()
