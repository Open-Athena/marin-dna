#!/usr/bin/env python3
"""Score the exact issue #402 test variants with mammalian phyloP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import polars as pl

from marin_dna.pipelines.evals.conservation import (
    REQUIRED_VARIANT_COLUMNS,
    SGE_VARIANT_COLUMNS,
    aggregate_conservation_metrics,
    aggregate_conservation_sge_metrics,
    score_variants_at_positions,
)

BENCHMARKS = ("mendelian_traits", "complex_traits", "sge")
SCORE_NAME = "phyloP_447m"
VARIANT_KEY = ("chrom", "pos", "ref", "alt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-root", type=Path, required=True)
    parser.add_argument("--bigwig", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    return parser.parse_args()


def _load_exact_variants(variants_root: Path, benchmark: str) -> pd.DataFrame:
    """Load metadata from one issue #402 final-checkpoint score table."""
    path = variants_root / benchmark / "variants.parquet"
    assert path.is_file(), f"missing exact variant table: {path}"
    frame = pl.read_parquet(path)
    required = REQUIRED_VARIANT_COLUMNS if benchmark != "sge" else SGE_VARIANT_COLUMNS
    assert set(required) <= set(frame.columns)
    assert frame.height > 0
    assert frame.select(VARIANT_KEY).n_unique() == frame.height
    assert frame.filter(
        pl.any_horizontal(pl.col(column).is_null() for column in required)
    ).is_empty()
    assert frame.filter(pl.col("pos") <= 0).is_empty(), (
        "input positions are VCF-style 1-based coordinates at this scoring boundary"
    )
    return frame.select(required).to_pandas()


def _score_benchmark(
    *,
    variants_root: Path,
    bigwig: Path,
    output_dir: Path,
    benchmark: str,
    n_bootstrap: int,
) -> dict[str, int]:
    """Score and aggregate one benchmark with the shared dashboard protocol."""
    variants = _load_exact_variants(variants_root, benchmark)
    scores = score_variants_at_positions(
        variants[list(VARIANT_KEY)],
        bigwig,
    )
    assert scores.shape == (len(variants),)
    scored = variants.copy()
    scored["score"] = scores
    benchmark_dir = output_dir / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    scored_path = benchmark_dir / "variants.parquet"
    scored.to_parquet(scored_path, index=False)

    if benchmark == "sge":
        metrics, report = aggregate_conservation_sge_metrics(
            {SCORE_NAME: scored_path},
            n_bootstrap=n_bootstrap,
            bootstrap_seed=0,
        )
    else:
        metrics, report = aggregate_conservation_metrics(
            {SCORE_NAME: scored_path},
            n_bootstrap=n_bootstrap,
            bootstrap_seed=0,
        )
    assert not metrics.empty
    metrics["split"] = "test"
    metrics["dataset"] = benchmark
    metrics.to_parquet(benchmark_dir / "metrics.parquet", index=False)
    (benchmark_dir / "report.md").write_text(report)
    return {
        "n_variants": len(scored),
        "n_unique_positions": int(scored[["chrom", "pos"]].drop_duplicates().shape[0]),
        "n_nan": int(scored["score"].isna().sum()),
    }


def main() -> None:
    args = parse_args()
    assert len(args.code_revision) == 40
    assert args.n_bootstrap > 0
    assert args.bigwig.is_file()
    assert args.bigwig.stat().st_size == 10_022_801_463
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_stats = {
        benchmark: _score_benchmark(
            variants_root=args.variants_root,
            bigwig=args.bigwig,
            output_dir=args.output_dir,
            benchmark=benchmark,
            n_bootstrap=args.n_bootstrap,
        )
        for benchmark in BENCHMARKS
    }
    manifest = {
        "analysis": "issue402 exact-row mammalian phyloP baseline",
        "score_name": SCORE_NAME,
        "track_source": args.source_uri,
        "track_size_bytes": args.bigwig.stat().st_size,
        "variant_source": (
            "46M final-checkpoint variants.parquet; exact benchmark metadata are "
            "model-independent and shared with the 104M evaluation"
        ),
        "split": "test",
        "coordinate_boundary": (
            "input pos is 1-based VCF convention; score_variants_at_positions "
            "converts to 0-based half-open [pos-1, pos)"
        ),
        "score_semantics": (
            "raw signed phyloP track score; NaN (unaligned) filled with 0 before metrics"
        ),
        "metrics": (
            "shared conservation matched-pair/SGE aggregators, "
            f"{args.n_bootstrap} bootstrap iterations, seed 0"
        ),
        "code_revision": args.code_revision,
        "benchmarks": benchmark_stats,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
