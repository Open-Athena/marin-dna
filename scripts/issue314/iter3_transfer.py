"""issue #314 iter3 — variant-type transfer (one model), nested-C.

Does a probe trained on one consequence transfer to others? Does a single pooled-all probe
match per-subset probes? Cross-subset, chromosome-grouped, leak-proof, and **with the
regularization tuned the same way as iter2** (no fixed-C confound): for train-arm A, each
chromosome c is predicted by a probe whose C is chosen by inner GroupKFold CV on A's
variants in the *other* chromosomes (nested LOCO), then applied to every variant on c. From
that all-variant OOF we read each eval subset's **per-chromosome-weighted AUPRC** and an
``all`` column = the **global AUPRC over all subsets bundled together**.

Rows = train-on subset A (+ ``pooled_all``); the ``LLR`` rows give the zero-shot baseline
(per-subset per-chrom + global ``all``). Fixed rep ``entire_window/abs_delta`` (symmetric →
valid across subsets). Saves a long-form parquet.
"""

import argparse

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iter1_representation_sweep import build_feature, load_and_pool

# mendelian is directional (ref=WT) → signed delta (the parsimony rule: signed feature for
# signed datasets, abs_delta for swap-invariant ones like complex/QTL).
REP = ("pool", "entire_window", "delta")
C_GRID = np.logspace(-12, 2, 10)  # nested-tuned per fold (matches iter2's wide/heavy grid)
MIN_VARIANTS = 100


def per_chrom_weighted_ap(y: np.ndarray, score: np.ndarray, chrom: np.ndarray) -> float:
    tot, w = 0.0, 0
    for c in np.unique(chrom):
        m = (chrom == c) & np.isfinite(score)
        if 0 < int(y[m].sum()) < int(m.sum()):
            tot += average_precision_score(y[m], score[m]) * int(m.sum())
            w += int(m.sum())
    return tot / w if w else float("nan")


def transfer_oof(fA, yA, cA, feat_all, chrom_all):
    """OOF over ALL variants: each chromosome is predicted by a probe trained on train-arm
    A's variants in the *other* chromosomes, with C chosen by inner GroupKFold CV on A."""
    oof = np.full(len(chrom_all), np.nan)
    for c in np.unique(chrom_all):
        tr = cA != c
        if not (0 < int(yA[tr].sum()) < int(tr.sum())):
            continue
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(l1_ratio=0.0, max_iter=2000))])
        k = min(5, len(np.unique(cA[tr])))
        if k >= 2:
            est = GridSearchCV(pipe, {"clf__C": C_GRID}, cv=GroupKFold(k),
                               scoring="average_precision", n_jobs=-1)
            est.fit(fA[tr], yA[tr], groups=cA[tr])
        else:  # too few train chromosomes to inner-CV — fall back to a strong fixed C
            est = pipe.set_params(clf__C=1e-3)
            est.fit(fA[tr], yA[tr])
        oof[chrom_all == c] = est.predict_proba(feat_all[chrom_all == c])[:, 1]
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
    print(f"{args.model}: {len(subsets)} subsets — nested-C transfer ({REP[1]}/{REP[2]})")

    rows = []

    def add(train: str, oof: np.ndarray) -> None:
        for b in subsets:  # per eval subset: per-chromosome-weighted AUPRC
            mb = masks[b]
            rows.append({"model": args.model, "train": train, "eval": b,
                         "auprc": per_chrom_weighted_ap(y_all[mb], oof[mb], chrom_all[mb])})
        fin = np.isfinite(oof)  # 'all' = global AUPRC over all subsets bundled together
        rows.append({"model": args.model, "train": train, "eval": "all",
                     "auprc": float(average_precision_score(y_all[fin], oof[fin]))})

    add("LLR", llr_all)  # zero-shot baseline (per-chrom per subset; global for 'all')
    train_arms = {s: masks[s] for s in subsets}
    train_arms["pooled_all"] = np.ones(len(y_all), dtype=bool)
    for a, ma in train_arms.items():
        oof = transfer_oof(feat[ma], y_all[ma], chrom_all[ma], feat, chrom_all)
        add(a, oof)
        cells = [r for r in rows if r["train"] == a]
        print(f"  train={a:12s} " + "  ".join(f"{c['eval'][:8]}:{c['auprc']:.3f}" for c in cells))

    pl.DataFrame(rows).write_parquet(f"{args.out}/transfer_{args.model}.parquet")
    print(f"\nwrote {args.out}/transfer_{args.model}.parquet")


if __name__ == "__main__":
    main()
