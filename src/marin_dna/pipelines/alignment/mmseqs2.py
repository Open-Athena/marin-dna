"""Project genomic intervals across genomes via mmseqs2 nucleotide search."""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Columns emitted by
#   mmseqs convertalis --format-output "query,target,tstart,tend,bits,evalue,fident,qcov,tcov"
# in the dELS_orthologs pipeline (snakemake/analysis/dELS_orthologs, issue #120).
# `tstart` / `tend` are mmseqs2's 1-based target coords; on reverse-strand hits
# `tend < tstart`. `bits` is the bit score used for best-hit-per-query ranking.
MMSEQS2_HITS_COLS: list[str] = [
    "query",
    "target",
    "tstart",
    "tend",
    "bits",
    "evalue",
    "fident",
    "qcov",
    "tcov",
]

MMSEQS2_HITS_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "query": pl.Utf8,
    "target": pl.Utf8,
    "tstart": pl.Int64,
    "tend": pl.Int64,
    "bits": pl.Float64,
    "evalue": pl.Float64,
    "fident": pl.Float64,
    "qcov": pl.Float64,
    "tcov": pl.Float64,
}

# Output of `project_hits_to_intervals`: 0-based half-open target coords plus
# the bit score carried through so best_hit_per_query can rank without a
# second pass.
PROJECTED_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "query": pl.Utf8,
    "chrom": pl.Utf8,
    "start": pl.Int64,
    "end": pl.Int64,
    "rev_strand": pl.Boolean,
    "bits": pl.Float64,
}


def parse_mmseqs2_hits(path: str | Path) -> pl.DataFrame:
    """Parse an mmseqs2 convertalis TSV into a polars DataFrame.

    Expects headerless TSV with columns listed in `MMSEQS2_HITS_COLS`. Empty
    files return an empty DataFrame with the canonical schema.
    """
    path = Path(path)
    if path.stat().st_size == 0:
        return pl.DataFrame(schema=MMSEQS2_HITS_SCHEMA)
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=MMSEQS2_HITS_COLS,
        schema_overrides=MMSEQS2_HITS_SCHEMA,
    )


def project_hits_to_intervals(hits: pl.DataFrame) -> pl.DataFrame:
    """Normalize mmseqs2 hits to 0-based half-open `(chrom, start, end)`.

    mmseqs2's `tstart`/`tend` are 1-based; on reverse-strand hits `tend <
    tstart`. We lift by `start = min(tstart, tend) - 1`, `end = max(tstart,
    tend)` and flag strand in `rev_strand`. The target FASTA record name is
    taken verbatim as `chrom` — callers must ensure the target DB is built
    from a FASTA whose record IDs are the chromosome IDs they want in the
    output parquet.
    """
    if hits.height == 0:
        return pl.DataFrame(schema=PROJECTED_SCHEMA)
    return hits.select(
        pl.col("query"),
        pl.col("target").alias("chrom"),
        (pl.min_horizontal("tstart", "tend") - 1).alias("start"),
        pl.max_horizontal("tstart", "tend").alias("end"),
        (pl.col("tend") < pl.col("tstart")).alias("rev_strand"),
        pl.col("bits"),
    )


def best_hit_per_query(projected: pl.DataFrame) -> pl.DataFrame:
    """Keep the highest-`bits` hit per query, one row per query.

    Ties are broken arbitrarily (polars `unique(keep="first")` after a
    descending sort on `bits`). Expects the output schema of
    `project_hits_to_intervals`.
    """
    if projected.height == 0:
        return projected
    return (
        projected.sort("bits", descending=True)
        .unique(subset=["query"], keep="first")
        .sort(["chrom", "start", "end"])
    )
