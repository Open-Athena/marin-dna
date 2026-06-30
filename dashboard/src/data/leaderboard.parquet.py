"""Observable Framework data loader: S3 metrics → one tidy parquet.

Pulls per-(method, dataset, subset) metric rows from S3 via
``marin_dna.pipelines.evals.leaderboard.normalized_rows``, prepends a
``dataset`` column, concatenates across datasets, and writes the resulting
DataFrame as a parquet blob on stdout for the dashboard to read via DuckDB.

Emits one parquet covering the matched-pair leaderboard datasets (mendelian_traits,
complex_traits). Each page filters by ``dataset``, so they coexist in one file. eQTL
was retired in PR #194 (issue #172); the caQTL/dsQTL zero-shot path was retired in #312
(the supervised accessibility-QTL benchmark + its dashboard page were removed in #332).
"""

from __future__ import annotations

import sys

import polars as pl

from marin_dna.pipelines.evals.leaderboard import normalized_rows

LEADERBOARD_DATASETS: tuple[str, ...] = (
    "mendelian_traits",
    "complex_traits",
)


def main() -> None:
    parts = []
    for dataset in LEADERBOARD_DATASETS:
        df = normalized_rows(dataset).with_columns(dataset=pl.lit(dataset))
        parts.append(df)
    out = pl.concat(parts, how="vertical_relaxed")
    out.write_parquet(sys.stdout.buffer)


if __name__ == "__main__":
    main()
