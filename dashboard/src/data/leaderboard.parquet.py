"""Observable Framework data loader: S3 metrics → one tidy parquet.

Pulls per-(method, dataset, subset) metric rows from S3 via
``marin_dna_evals.leaderboard.normalized_rows`` (zero-shot likelihood metrics)
and ``probe_normalized_rows`` (the frozen-embedding linear probe, #347/#348), tags each
with a ``dataset`` and a ``supervision`` column (``unsupervised`` / ``supervised``),
concatenates, and writes the resulting DataFrame as a parquet blob on stdout for the
dashboard to read via DuckDB. The Mendelian page's top-level mode toggle filters on
``supervision`` so the two metric-worlds are never mixed in one ranked view.

Emits one parquet covering the matched-pair leaderboard datasets (mendelian_traits,
complex_traits). Each page filters by ``dataset``, so they coexist in one file. Supervised
rows exist where a probe-capable model (marin_dna on S3, Evo 2 on the pinned gist) has probe
metrics (mendelian today). eQTL
was retired in PR #194 (issue #172); the caQTL/dsQTL zero-shot path was retired in #312
(the supervised official-metrics benchmark now lives on its own Accessibility QTL page,
fed by ``accessibility_qtl.parquet.py``).
"""

from __future__ import annotations

import sys

import polars as pl
from marin_dna_evals.leaderboard import (
    normalized_rows,
    probe_normalized_rows,
)

LEADERBOARD_DATASETS: tuple[str, ...] = (
    "mendelian_traits",
    "complex_traits",
)

# Datasets that also have frozen-embedding probe metrics on S3 (the Supervised view). Only
# mendelian today; extend as models are probed on other datasets. Kept explicit so the build
# doesn't issue a doomed S3 read (one 404 per marin_dna model) for datasets with no probes.
PROBE_DATASETS: tuple[str, ...] = ("mendelian_traits",)


def main() -> None:
    parts = [
        normalized_rows(dataset).with_columns(
            dataset=pl.lit(dataset), supervision=pl.lit("unsupervised")
        )
        for dataset in LEADERBOARD_DATASETS
    ]
    # Supervised (linear-probe) rows, tagged with the same `supervision` column the Mendelian
    # page's mode toggle filters on. probe_normalized_rows returns an explicitly-typed frame
    # (empty if a dataset has no probes yet), so it concatenates cleanly.
    parts += [
        probe_normalized_rows(dataset).with_columns(
            dataset=pl.lit(dataset), supervision=pl.lit("supervised")
        )
        for dataset in PROBE_DATASETS
    ]
    out = pl.concat(parts, how="vertical_relaxed")
    out.write_parquet(sys.stdout.buffer)


if __name__ == "__main__":
    main()
