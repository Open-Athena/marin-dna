"""issue #270 (cLLR subsampling pilot): score the pilot variants' LLR on a GPU.

Thin wrapper over the eval LLR bundle (`compute_variant_scores`), FWD+RC
averaged — identical kernel/convention to evals_v2. Adds `llr_avg` and keeps the
pentanuc columns so the convergence step can group by cell.

Fork-safety (matches evals_v2): this process must NOT initialize s3fs/fsspec in
its *parent* before the DataLoader forks, or the workers inherit fsspec's dead
async loop and deadlock on the lazy S3 genome reads. So checkpoint + variants are
**local** here (staged by `issue270_fetch.py` in a separate process), and the
ONLY S3 access is the genome — opened lazily *inside each worker* by `Genome`,
exactly as evals_v2 does it. Output is written locally; `issue270_fetch.py`
uploads it.

Run (on the GPU node, after `issue270_fetch.py --mode download`):
    uv run python scripts/issue270_score_llr.py \
        --checkpoint ckpt --variants pilot_variants.parquet --out pilot_scores.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from marin_dna.pipelines.evals.inference import compute_variant_scores

# Canonical GRCh38 reference in S3 — read by byte-range via pyfaidx + s3fs,
# lazily inside each DataLoader worker (no parent-process fsspec init). Same
# path evals_v2 uses.
GENOME_S3 = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt", help="LOCAL HF checkpoint dir")
    ap.add_argument(
        "--variants", default="pilot_variants.parquet", help="LOCAL parquet"
    )
    ap.add_argument(
        "--out", default="pilot_scores.parquet", help="LOCAL output parquet"
    )
    ap.add_argument("--genome", default=GENOME_S3, help="S3 genome (lazy, in workers)")
    ap.add_argument("--context-size", type=int, default=255)  # exp135: 255 bp + BOS
    ap.add_argument("--batch-size", type=int, default=128)  # validated A10G default
    args = ap.parse_args()

    # LOCAL read -> no parent-process fsspec init (preserves fork safety).
    variants = pd.read_parquet(args.variants)
    print(
        f"variants: {len(variants):,} rows; "
        f"cells: {variants['pentanuc_mut'].nunique()}; "
        f"contexts: {sorted(variants['context_label'].unique())}"
    )

    scores = compute_variant_scores(
        checkpoint_path=args.checkpoint,  # LOCAL
        dataset=variants[["chrom", "pos", "ref", "alt"]],
        genome_path=args.genome,  # S3, opened lazily per worker
        context_size=args.context_size,
        batch_size=args.batch_size,
        num_workers=4,
        data_transform_on_the_fly=True,
        torch_compile=True,
        rc=True,
    )
    assert len(scores) == len(variants), (len(scores), len(variants))
    assert {"llr_fwd", "llr_rc"} <= set(scores.columns), scores.columns.tolist()

    out = pd.concat(
        [variants.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )
    out["llr_avg"] = (out["llr_fwd"] + out["llr_rc"]) / 2.0
    assert out[["llr_fwd", "llr_rc", "llr_avg"]].notna().all().all(), (
        "NaN in LLR columns"
    )

    out.to_parquet(args.out, index=False)  # LOCAL; uploaded by issue270_fetch.py
    print(f"wrote {len(out):,} scored variants -> {args.out}")
    print("\nmean llr_avg per cell:")
    print(
        out.groupby(["context_label", "pentanuc_mut"])["llr_avg"]
        .agg(["mean", "std", "count"])
        .to_string()
    )


if __name__ == "__main__":
    main()
