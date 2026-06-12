"""issue #314 — dsQTL accessibility probe (EDA, exp166-v0.1-p1B).

dsQTL is condition-specific chromatin-accessibility QTL — orthogonal to the
fitness/conservation signal a gLM log-likelihood captures — so this asks whether the
**frozen embedding alone** linearly predicts dsQTL significance, where the LLR is not
expected to help. Symmetric features only (QTL has no ref/alt polarity), FWD+RC
averaged, chromosome-grouped OOF, **global** AUPRC vs the random floor (~0.02) and the
supervised ChromBPNet/Enformer ceilings (~0.43–0.54; TraitGym / #310).

EDA caveats (not the fair protocol): C is picked as best-of-grid on the same OOF
(optimistic), scaler-on + no-PCA (the iter1 parsimony recipe), and the global metric
pools predictions across fold-models (cross-fold calibration caveat). The fair
nested-LOCO + per-chrom metric is deferred.
"""

import argparse

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from iter1_representation_sweep import _rep_label, build_feature, load_and_pool
from marin_dna.pipelines.evals.variant_probe import chrom_grouped_oof

# symmetric reps only — dsQTL has no ref/alt direction
REPS = [
    ("pool", "entire_window", "abs_delta"),
    ("pool", "entire_window", "sum_absdiff"),
    ("pool", "center", "abs_delta"),
    ("innerprod",),  # TraitGym's headline embedding feature
    ("cov_delta",),  # EVEE covariance pooling
]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CLASS_WEIGHTS: list[str | None] = [None, "balanced"]  # the minority-reweighting A/B
CEILINGS = {"random": 0.020, "ChromBPNet ATAC": 0.538, "Enformer DNase": 0.526}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="s3:// dsQTL embedding cache")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    pooled, inner, cov, keys = load_and_pool(args.cache)
    kdf = keys.to_pandas()
    kdf["chrom"] = kdf["chrom"].astype(str)
    y = kdf["label"].to_numpy().astype(int)
    chrom = kdf["chrom"].to_numpy()
    print(
        f"\ndsQTL exp166-v0.1-p1B: n={len(y)} pos={int(y.sum())} "
        f"(rate {y.mean():.4f}) chroms={len(set(chrom))}"
    )
    print("ceilings: " + "  ".join(f"{k} {v:.3f}" for k, v in CEILINGS.items()))
    print(f"\n{'rep':26s} {'dim':>6}  {'cw=None':>16}  {'cw=balanced':>16}")
    rows = []
    for rep in REPS:
        feat = build_feature(pooled, inner, cov, rep)
        best: dict[str | None, tuple[float, float]] = {}
        for cw in CLASS_WEIGHTS:
            scan = []
            for c in C_GRID:
                oof = chrom_grouped_oof(
                    feat, y, chrom, loss="logistic", c=c,
                    n_pca=None, standardize=True, n_splits=args.n_splits,
                    class_weight=cw,
                )
                scan.append((c, float(average_precision_score(y, oof))))
            c_b, ap_b = max(scan, key=lambda t: t[1])
            best[cw] = (ap_b, c_b)
        (ap_n, c_n), (ap_b, c_b) = best[None], best["balanced"]
        print(
            f"{_rep_label(rep):26s} {feat.shape[1]:>6}  "
            f"{ap_n:.3f} (C={c_n:.0e})   {ap_b:.3f} (C={c_b:.0e})"
        )
        rows.append({"rep": _rep_label(rep), "dim": feat.shape[1],
                     "auprc_none": ap_n, "c_none": c_n,
                     "auprc_balanced": ap_b, "c_balanced": c_b})
    pl.DataFrame(rows).write_parquet(
        "s3://oa-bolinas/analysis/issue314/dsqtl_probe/exp166-v0.1-p1B.parquet"
    )
    print("\nwrote s3://oa-bolinas/analysis/issue314/dsqtl_probe/exp166-v0.1-p1B.parquet")


if __name__ == "__main__":
    main()
