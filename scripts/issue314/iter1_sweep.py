"""issue #314 iter1 — stratified random-search sweep, all Mendelian subsets, saving OOF.

Every representation (the scientific axis) is sampled, with random tuning knobs
(``StandardScaler`` on/off, PCA-k ∈ {8,32,64,128,256,none}, C ~ log-uniform
[1e-3,1e2]). Per (config × subset): chromosome-grouped 3-fold OOF → **save the OOF
predictions + the point AUPRC**. No bootstrap in the loop — significance and the
factorial attribution are fast post-hoc steps on the saved artifacts
(``iter1_attribution.py``). joblib threading + single-thread BLAS parallelizes the
cells over the shared in-memory pooled features.

Run:
  OMP_NUM_THREADS=1 uv run --group genome-s3 python scripts/issue314/iter1_sweep.py \
    --cache s3://oa-bolinas/analysis/issue314/embeddings/exp135-1B-m5.1/mendelian_traits \
    --out s3://oa-bolinas/analysis/issue314/iter1_search/exp135-1B-m5.1
"""

import argparse

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from iter1_representation_sweep import _rep_label, build_feature, load_and_pool
from marin_dna.pipelines.evals.variant_probe import (
    PAIR_COMBOS,
    POOLING_EXTENTS,
    chrom_grouped_oof,
)

REPS = [("pool", e, c) for e in POOLING_EXTENTS for c in PAIR_COMBOS] + [
    ("innerprod",),
    ("cov_delta",),
]
PCA_CHOICES: list[int | None] = [8, 32, 64, 128, 256, None]
MIN_VARIANTS = 100  # skip tiny subsets (chrom-grouped CV is unstable below this)


def sample_configs(n_per_rep: int, seed: int) -> list[dict]:
    """Stratified random search: every rep sampled ``n_per_rep`` times with random
    tuning knobs (scaler ~ Bernoulli, PCA-k ~ uniform, C ~ log-uniform[1e-3,1e2])."""
    rng = np.random.default_rng(seed)
    cfgs = []
    for rep in REPS:
        for _ in range(n_per_rep):
            cfgs.append(
                {
                    "rep": rep,
                    "scaler": bool(rng.integers(0, 2)),
                    "n_pca": PCA_CHOICES[int(rng.integers(0, len(PCA_CHOICES)))],
                    "c": float(10.0 ** rng.uniform(-3, 2)),
                }
            )
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="s3:// embedding cache prefix")
    ap.add_argument("--out", required=True, help="s3:// output prefix")
    ap.add_argument("--n_per_rep", type=int, default=10)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pooled, inner, cov, keys = load_and_pool(args.cache)
    kdf = keys.to_pandas()
    kdf["chrom"] = kdf["chrom"].astype(str)
    y_all = kdf["label"].to_numpy().astype(int)
    chrom_all = kdf["chrom"].to_numpy()
    subset_all = kdf["subset"].to_numpy()

    print("building rep features (FWD+RC averaged)...")
    rep_feats = {
        _rep_label(r): build_feature(pooled, inner, cov, r) for r in tqdm(REPS)
    }
    del pooled, inner, cov

    subsets = [
        s
        for s in sorted(set(subset_all))
        if (subset_all == s).sum() >= MIN_VARIANTS
        and len(set(chrom_all[subset_all == s])) >= 3
    ]
    # pre-slice features/labels per (rep, subset) once (shared read-only by threads)
    feat_sub, y_sub, chrom_sub = {}, {}, {}
    for s in subsets:
        m = subset_all == s
        y_sub[s], chrom_sub[s] = y_all[m], chrom_all[m]
        for rl, f in rep_feats.items():
            feat_sub[(rl, s)] = f[m]
    del rep_feats

    configs = sample_configs(args.n_per_rep, args.seed)
    cells = [(s, ci) for s in subsets for ci in range(len(configs))]
    print(
        f"{len(configs)} configs × {len(subsets)} subsets = {len(cells)} cells "
        f"(3-fold OOF, {args.n_jobs} threads)"
    )

    def cell(s: str, ci: int) -> tuple[str, int, np.ndarray, float]:
        cfg = configs[ci]
        oof = chrom_grouped_oof(
            feat_sub[(_rep_label(cfg["rep"]), s)],
            y_sub[s],
            chrom_sub[s],
            loss="logistic",
            c=cfg["c"],
            n_pca=cfg["n_pca"],
            standardize=cfg["scaler"],
            n_splits=3,
        )
        return s, ci, oof, float(average_precision_score(y_sub[s], oof))

    results = Parallel(n_jobs=args.n_jobs, backend="threading", verbose=10)(
        delayed(cell)(s, ci) for s, ci in cells
    )

    rows, oof_by_subset = [], {s: {} for s in subsets}
    for s, ci, oof, auprc in results:
        cfg = configs[ci]
        rep = cfg["rep"]
        rows.append(
            {
                "subset": s,
                "config_id": ci,
                "rep": _rep_label(rep),
                "pooling": rep[1] if rep[0] == "pool" else "-",
                "combo": rep[2] if rep[0] == "pool" else rep[0],
                "rep_kind": "pool" if rep[0] == "pool" else rep[0],
                "scaler": cfg["scaler"],
                "n_pca": cfg["n_pca"] if cfg["n_pca"] is not None else 0,
                "c": cfg["c"],
                "auprc": auprc,
            }
        )
        oof_by_subset[s][f"oof_{ci}"] = oof

    pl.DataFrame(rows).write_parquet(f"{args.out}/auprc.parquet")
    key_cols = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "match_group",
        "llr_fwd",
        "llr_rc",
    ]
    for s in subsets:
        m = subset_all == s
        sub_keys = pl.from_pandas(kdf.loc[m, key_cols].reset_index(drop=True))
        oof_df = pl.DataFrame(oof_by_subset[s])
        pl.concat([sub_keys, oof_df], how="horizontal").write_parquet(
            f"{args.out}/oof_{s}.parquet"
        )
    print(f"\nwrote {args.out}/auprc.parquet + {len(subsets)} per-subset OOF parquets")


if __name__ == "__main__":
    main()
