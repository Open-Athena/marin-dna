"""issue #314 iter1 — representation sweep on the frozen-embedding cache.

Streams the ``{fwd,rc}×{ref,alt}`` per-token cache (sharded float16), pools every
extent + computes the per-strand ``innerprod`` / ``cov_delta`` cross-features,
FWD+RC-averages the per-strand pair features, and runs the chromosome-grouped OOF
logistic probe per representation — scored by AUPRC vs the zero-shot LLR baseline
that rides along in the keys. Mendelian, one subset at a time (missense first).

Holds RAM bounded by pooling each shard on load (the 63 GB cache is streamed, not
materialized); only the pooled ``[N, D]`` features accumulate.

Run:
  uv run --group genome-s3 python scripts/issue314/iter1_representation_sweep.py \
    --cache s3://oa-bolinas/analysis/issue314/embeddings/exp135-1B-m5.1/mendelian_traits \
    --subset missense_variant
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import polars as pl
import s3fs
from tqdm import tqdm

from marin_dna.pipelines.evals.metrics import (
    auprc_with_bootstrap_se,
    paired_metric_delta_bootstrap,
)
from marin_dna.pipelines.evals.variant_probe import (
    PAIR_COMBOS,
    POOLING_EXTENTS,
    chrom_grouped_oof,
    cov_delta_feature,
    innerprod_feature,
    pair_feature,
    pool_tokens,
    random_projection,
)

# Variant DNA position in the cache: in_seq_var_pos(255) = 127 on both strands
# (the BOS prefix is already dropped in the extraction kernel).
VAR_INDEX = 127


def _read_shard(fs: s3fs.S3FileSystem, npz: str) -> tuple[np.ndarray, pl.DataFrame]:
    # Stream np.load straight off the s3fs file — reading to bytes + wrapping in
    # BytesIO would hold ~3x the array per in-flight read and OOM the readers.
    with fs.open(f"s3://{npz}") as fh:
        emb = np.load(fh)["emb"]  # float16, [chunk, 2, 2, L, D]
    keys = pl.read_parquet(f"s3://{npz}".replace(".npz", ".keys.parquet"))
    return emb, keys


def load_and_pool(
    cache: str, *, n_center: int = 100, cov_r: int = 64, n_readers: int = 4
) -> tuple[dict, dict, dict, pl.DataFrame]:
    """Stream shards; return pooled ``(strand,allele,extent) -> [N,D]``, per-strand
    ``innerprod``/``cov_delta`` ``strand -> [N,·]``, and the concatenated keys.

    Shards are read in parallel batches (single-stream S3 is the bottleneck) and
    pooled on arrival, so RAM stays bounded to ~``n_readers`` shards at a time.
    """
    fs = s3fs.S3FileSystem()
    npzs = sorted(f for f in fs.ls(cache) if f.endswith(".npz"))
    assert npzs, f"no shards under {cache}"
    pools: dict[tuple[int, int, str], list[np.ndarray]] = {}
    inner: dict[int, list[np.ndarray]] = {0: [], 1: []}
    cov: dict[int, list[np.ndarray]] = {0: [], 1: []}
    keys: list[pl.DataFrame] = []
    proj: np.ndarray | None = None
    with ThreadPoolExecutor(max_workers=n_readers) as ex:
        for i in tqdm(range(0, len(npzs), n_readers), desc="pool shard-batches"):
            batch = npzs[i : i + n_readers]
            for emb, keys_df in ex.map(partial(_read_shard, fs), batch):
                assert np.isfinite(emb[::16]).all(), "non-finite in a shard (sampled)"
                keys.append(keys_df)
                if proj is None:
                    proj = random_projection(emb.shape[-1], cov_r, seed=0)
                for s in (0, 1):  # 0=fwd, 1=rc
                    # Upcast each [chunk, L, D] slice to float32 (one slice at a
                    # time) so pooling accumulates in float32 cleanly.
                    ref_tok = emb[:, s, 0].astype(np.float32)
                    alt_tok = emb[:, s, 1].astype(np.float32)
                    for ext in POOLING_EXTENTS:
                        kw = dict(var_index=VAR_INDEX, n_center=n_center)
                        pools.setdefault((s, 0, ext), []).append(
                            pool_tokens(ref_tok, ext, **kw)
                        )
                        pools.setdefault((s, 1, ext), []).append(
                            pool_tokens(alt_tok, ext, **kw)
                        )
                    inner[s].append(innerprod_feature(ref_tok, alt_tok))
                    cov[s].append(cov_delta_feature(ref_tok, alt_tok, proj))
                del emb  # free the shard before the next
    pooled = {k: np.concatenate(v) for k, v in pools.items()}
    inner_c = {s: np.concatenate(inner[s]) for s in (0, 1)}
    cov_c = {s: np.concatenate(cov[s]) for s in (0, 1)}
    return pooled, inner_c, cov_c, pl.concat(keys)


def build_feature(pooled: dict, inner: dict, cov: dict, rep: tuple) -> np.ndarray:
    """FWD+RC-averaged feature for ``rep`` = ``('pool', ext, combo)`` | ``('innerprod',)``
    | ``('cov_delta',)``."""
    if rep[0] == "pool":
        _, ext, combo = rep
        f = [
            pair_feature(pooled[(s, 0, ext)], pooled[(s, 1, ext)], combo)
            for s in (0, 1)
        ]
        return (f[0] + f[1]) / 2
    if rep[0] == "innerprod":
        return (inner[0] + inner[1]) / 2
    if rep[0] == "cov_delta":
        return (cov[0] + cov[1]) / 2
    raise ValueError(rep)


def _rep_label(rep: tuple) -> str:
    return rep[0] if len(rep) == 1 else f"{rep[1]}/{rep[2]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache", required=True, help="s3:// prefix of the embedding cache"
    )
    ap.add_argument("--subset", default="missense_variant")
    ap.add_argument("--n_center", type=int, default=100)
    ap.add_argument("--n_pca", type=int, default=256)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--out", default="s3://oa-bolinas/analysis/issue314/iter1")
    args = ap.parse_args()

    pooled, inner, cov, keys = load_and_pool(args.cache, n_center=args.n_center)
    kdf = keys.to_pandas()
    kdf["chrom"] = kdf["chrom"].astype(str)
    mask = (kdf["subset"] == args.subset).to_numpy()
    n = int(mask.sum())
    y = kdf["label"].to_numpy().astype(int)[mask]
    chrom = kdf["chrom"].to_numpy()[mask]
    llr = (-(kdf["llr_fwd"] + kdf["llr_rc"]) / 2).to_numpy()[mask]  # minus_llr_avg
    mg = kdf["match_group"].to_numpy()[mask]
    ysr, mgsr, llrsr = pd.Series(y), pd.Series(mg), pd.Series(llr)
    llr_stat = auprc_with_bootstrap_se(ysr, llrsr, mgsr)
    print(
        f"\nsubset={args.subset}  n={n}  n_pos={int(y.sum())}  n_chrom={len(set(chrom))}  "
        f"n_match_groups={len(set(mg))}\n"
        f"probe: PCA-{args.n_pca} C={args.c} logistic, chrom-grouped OOF; AUPRC ± "
        f"cluster-bootstrap SE (resample match_group, 1000x); Δ vs LLR = paired "
        f"bootstrap [95% CI], two-sided p\n"
        f"zero-shot LLR (minus_llr_avg) AUPRC = {llr_stat['value']:.3f} ± {llr_stat['se']:.3f}"
    )

    reps = [("pool", ext, combo) for ext in POOLING_EXTENTS for combo in PAIR_COMBOS]
    reps += [("innerprod",), ("cov_delta",)]
    rows = []
    for rep in tqdm(reps, desc="probe reps"):
        feat = build_feature(pooled, inner, cov, rep)[mask]
        oof = chrom_grouped_oof(
            feat, y, chrom, n_pca=args.n_pca, c=args.c, standardize=True
        )
        oofsr = pd.Series(oof)
        a = auprc_with_bootstrap_se(ysr, oofsr, mgsr)
        d = paired_metric_delta_bootstrap(ysr, oofsr, llrsr, mgsr)
        rows.append(
            {
                "representation": _rep_label(rep),
                "dim": feat.shape[1],
                "auprc": a["value"],
                "auprc_se": a["se"],
                "delta_llr": d["delta"],
                "delta_se": d["se"],
                "ci_low": d["ci_low"],
                "ci_high": d["ci_high"],
                "p": d["p_two_sided"],
            }
        )

    rows.sort(key=lambda r: -r["auprc"])
    print(
        f"\n{'representation':26s} {'dim':>5s} {'AUPRC ± SE':>14s} "
        f"{'Δ vs LLR ± SE':>16s} {'p':>7s}"
    )
    print("-" * 72)
    for r in rows:
        sig = "*" if r["p"] < 0.05 else " "
        print(
            f"{r['representation']:26s} {r['dim']:5d}   {r['auprc']:.3f} ± {r['auprc_se']:.3f}"
            f"   {r['delta_llr']:+.3f} ± {r['delta_se']:.3f}  {r['p']:6.3f}{sig}"
        )

    out_path = f"{args.out}/{args.subset}.parquet"
    pl.DataFrame(rows).write_parquet(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
