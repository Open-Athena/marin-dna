"""Observable Framework data loader: S3 metrics → one tidy parquet.

Pulls per-(method, dataset, subset) metric rows from S3 via
``marin_dna.pipelines.evals.leaderboard.normalized_rows`` (zero-shot likelihood metrics)
and ``probe_normalized_rows`` (the frozen-embedding linear probe, #347/#348), tags each
with a ``dataset`` and a ``supervision`` column (``unsupervised`` / ``supervised``),
concatenates, and writes the resulting DataFrame as a parquet blob on stdout for the
dashboard to read via DuckDB. The Mendelian page's top-level mode toggle filters on
``supervision`` so the two metric-worlds are never mixed in one ranked view.

Emits one parquet covering the matched-pair leaderboard datasets (mendelian_traits,
complex_traits). Each page filters by ``dataset``, so they coexist in one file. Supervised
rows exist only where a marin_dna model has probe metrics on S3 (mendelian today). eQTL
was retired in PR #194 (issue #172); the caQTL/dsQTL zero-shot path was retired in #312
(the supervised official-metrics benchmark now lives on its own Accessibility QTL page,
fed by ``accessibility_qtl.parquet.py``).
"""

from __future__ import annotations

import sys

import polars as pl

from marin_dna.pipelines.evals.leaderboard import (
    normalized_rows,
    probe_normalized_rows,
)

LEADERBOARD_DATASETS: tuple[str, ...] = (
    "mendelian_traits",
    "complex_traits",
)


def main() -> None:
    parts = []
    for dataset in LEADERBOARD_DATASETS:
        parts.append(
            normalized_rows(dataset).with_columns(
                dataset=pl.lit(dataset), supervision=pl.lit("unsupervised")
            )
        )
        # Supervised (linear-probe) rows where a marin_dna model has #347-schema probe
        # metrics on S3 — mendelian only today, empty elsewhere. Appended only when present
        # so an empty frame never perturbs the concat schema.
        supervised = probe_normalized_rows(dataset).with_columns(
            dataset=pl.lit(dataset), supervision=pl.lit("supervised")
        )
        if supervised.height:
            parts.append(supervised)
    out = pl.concat(parts, how="vertical_relaxed")
    out.write_parquet(sys.stdout.buffer)


if __name__ == "__main__":
    main()
