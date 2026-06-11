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
import io

import numpy as np
import polars as pl
import s3fs
from sklearn.metrics import average_precision_score

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


def load_and_pool(
    cache: str, *, n_center: int = 100, cov_r: int = 64
) -> tuple[dict, dict, dict, pl.DataFrame]:
    """Stream shards; return pooled ``(strand,allele,extent) -> [N,D]``, per-strand
    ``innerprod``/``cov_delta`` ``strand -> [N,·]``, and the concatenated keys."""
    fs = s3fs.S3FileSystem()
    npzs = sorted(f for f in fs.ls(cache) if f.endswith(".npz"))
    assert npzs, f"no shards under {cache}"
    pools: dict[tuple[int, int, str], list[np.ndarray]] = {}
    inner: dict[int, list[np.ndarray]] = {0: [], 1: []}
    cov: dict[int, list[np.ndarray]] = {0: [], 1: []}
    keys: list[pl.DataFrame] = []
    proj: np.ndarray | None = None
    for npz in npzs:
        emb = np.load(io.BytesIO(fs.open(f"s3://{npz}").read()))["emb"]  # float16
        assert np.isfinite(emb).all(), f"non-finite values in {npz}"
        keys.append(pl.read_parquet(f"s3://{npz}".replace(".npz", ".keys.parquet")))
        if proj is None:
            proj = random_projection(emb.shape[-1], cov_r, seed=0)
        for s in (0, 1):  # 0=fwd, 1=rc
            # Upcast each [chunk, L, D] slice to float32 (one slice at a time, not
            # the whole 2 GB shard) so pooling accumulates in float32 cleanly.
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
        del emb  # free the 2 GB shard before streaming the next
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
    args = ap.parse_args()

    pooled, inner, cov, keys = load_and_pool(args.cache, n_center=args.n_center)
    kdf = keys.to_pandas()
    kdf["chrom"] = kdf["chrom"].astype(str)
    mask = (kdf["subset"] == args.subset).to_numpy()
    n = int(mask.sum())
    y = kdf["label"].to_numpy().astype(int)[mask]
    chrom = kdf["chrom"].to_numpy()[mask]
    llr = (-(kdf["llr_fwd"] + kdf["llr_rc"]) / 2).to_numpy()[mask]  # minus_llr_avg
    print(
        f"subset={args.subset}  n={n}  n_pos={int(y.sum())}  n_chrom={len(set(chrom))}  "
        f"probe: PCA-{args.n_pca} C={args.c} logistic, chrom-grouped OOF"
    )
    ap_llr = average_precision_score(y, llr)

    reps = [("pool", ext, combo) for ext in POOLING_EXTENTS for combo in PAIR_COMBOS]
    reps += [("innerprod",), ("cov_delta",)]
    rows = []
    for rep in reps:
        feat = build_feature(pooled, inner, cov, rep)[mask]
        oof = chrom_grouped_oof(
            feat, y, chrom, n_pca=args.n_pca, c=args.c, standardize=True
        )
        rows.append((_rep_label(rep), feat.shape[1], average_precision_score(y, oof)))

    rows.sort(key=lambda r: -r[2])
    print(f"\n{'representation':28s} {'dim':>6s} {'AUPRC':>7s} {'vs LLR':>8s}")
    print(f"{'zero-shot LLR (minus_llr_avg)':28s} {'-':>6s} {ap_llr:7.3f} {'—':>8s}")
    print("-" * 53)
    for label, dim, auprc in rows:
        print(f"{label:28s} {dim:6d} {auprc:7.3f} {auprc - ap_llr:+8.3f}")

    out = pl.DataFrame(
        {
            "representation": [r[0] for r in rows],
            "dim": [r[1] for r in rows],
            "auprc": [r[2] for r in rows],
            "auprc_minus_llr": [r[2] - ap_llr for r in rows],
        }
    )
    out_path = f"scratch/issue314_iter1_{args.subset}.parquet"
    out.write_parquet(out_path)
    print(f"\nwrote {out_path}  (LLR baseline AUPRC = {ap_llr:.3f})")


if __name__ == "__main__":
    main()
