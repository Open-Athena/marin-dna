"""issue #314 iter2 — TraitGym nested-LOCO protocol (per-model, leakage-free C tuning).

Resolves the fixed-recipe confound from iter1: holds the representation fixed across models
(fair) and re-tunes C per model and per fold via nested leave-one-chromosome-out (fair across
dimensionality). Reps: ``entire_window/abs_delta`` (symmetric — universal, good for the
zoonomia distal where the fixed center/sum_absdiff failed) and ``entire_window/delta``
(signed — the directional arm for Mendelian). Per (subset, rep): global AUPRC (match_group
cluster-bootstrap, our convention) + per-chromosome-weighted AUPRC (TraitGym's metric) + Δ vs
zero-shot LLR (paired bootstrap) + the selected-C range across folds (verify the grid isn't
truncating any model — every optimum should be interior). Saves a parquet.

Run:
  uv run --group genome-s3 python scripts/issue314/iter2_nested.py \
    --cache s3://.../embeddings/<model>/mendelian_traits --out s3://.../iter2_nested --model <model>
"""

import argparse

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from iter1_representation_sweep import _rep_label, build_feature, load_and_pool
from marin_dna.pipelines.evals.metrics import (
    auprc_with_bootstrap_se,
    paired_metric_delta_bootstrap,
)
from marin_dna.pipelines.evals.variant_probe import traitgym_nested_oof

REPS = [("pool", "entire_window", "abs_delta"), ("pool", "entire_window", "delta")]
C_GRID = np.logspace(-8, 2, 12)  # wide + heavy; selected-C range reported to check truncation
MIN_VARIANTS = 100


def per_chrom_weighted_ap(y: np.ndarray, score: np.ndarray, chrom: np.ndarray) -> float:
    """TraitGym metric: AUPRC within each chromosome, sample-size-weighted mean
    (chromosomes lacking both classes are skipped)."""
    tot, w = 0.0, 0
    for c in np.unique(chrom):
        m = chrom == c
        if 0 < int(y[m].sum()) < int(m.sum()):
            tot += average_precision_score(y[m], score[m]) * int(m.sum())
            w += int(m.sum())
    return tot / w if w else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    pooled, inner, cov, keys = load_and_pool(args.cache)
    kdf = keys.to_pandas()
    kdf["chrom"] = kdf["chrom"].astype(str)
    y_all = kdf["label"].to_numpy().astype(int)
    chrom_all = kdf["chrom"].to_numpy()
    subset_all = kdf["subset"].to_numpy()
    mg_all = kdf["match_group"].to_numpy()
    llr_all = (-(kdf["llr_fwd"] + kdf["llr_rc"]) / 2).to_numpy()  # minus_llr_avg
    feats = {_rep_label(r): build_feature(pooled, inner, cov, r) for r in REPS}
    del pooled, inner, cov

    subsets = [
        s for s in sorted(set(subset_all))
        if (subset_all == s).sum() >= MIN_VARIANTS
        and len(set(chrom_all[subset_all == s])) >= 3
    ]
    print(f"{args.model}: {len(subsets)} subsets, nested LOGO + inner GridSearchCV(C)")
    rows = []
    for s in subsets:
        m = subset_all == s
        y, chrom, mg, llr = y_all[m], chrom_all[m], mg_all[m], llr_all[m]
        llr_g = auprc_with_bootstrap_se(y, llr, mg, rng=0)
        llr_pc = per_chrom_weighted_ap(y, llr, chrom)
        for rl, f in feats.items():
            try:
                oof, sel = traitgym_nested_oof(f[m], y, chrom, c_grid=C_GRID, inner_splits=5)
            except Exception as e:  # keep the unattended run alive on a bad cell
                print(f"  SKIP {s} {rl}: {type(e).__name__} {e}")
                continue
            g = auprc_with_bootstrap_se(y, oof, mg, rng=0)
            pc = per_chrom_weighted_ap(y, oof, chrom)
            d = paired_metric_delta_bootstrap(y, oof, llr, mg, rng=0)
            sig = "✓" if d["p_two_sided"] < 0.05 and d["delta"] > 0 else ""
            print(
                f"  {s:22s} {rl:24s} probe {g['value']:.3f}±{g['se']:.3f} "
                f"(perchrom {pc:.3f}) | LLR {llr_g['value']:.3f} (perchrom {llr_pc:.3f}) "
                f"| Δ {d['delta']:+.3f} p={d['p_two_sided']:.3f}{sig} "
                f"| C 10^[{np.log10(min(sel)):+.1f},{np.log10(max(sel)):+.1f}] "
                f"med {np.median(sel):.1e}"
            )
            rows.append({
                "model": args.model, "subset": s, "rep": rl, "n": int(m.sum()),
                "probe_global": g["value"], "probe_se": g["se"], "probe_perchrom": pc,
                "llr_global": llr_g["value"], "llr_perchrom": llr_pc,
                "delta": d["delta"], "delta_se": d["se"], "delta_p": d["p_two_sided"],
                "c_med": float(np.median(sel)), "c_min": float(min(sel)),
                "c_max": float(max(sel)),
            })
    pl.DataFrame(rows).write_parquet(f"{args.out}/nested_{args.model}.parquet")
    print(f"\nwrote {args.out}/nested_{args.model}.parquet")


if __name__ == "__main__":
    main()
