"""Post-projection filters for halLiftover output.

A query window can split into multiple lift records when it crosses
alignment block boundaries. :func:`filter_single_chrom_strand` groups by
``(query_name, species)``, drops groups that disagree on chrom or strand,
and merges the remaining group into one min/max span. A subsequent
:func:`filter_length` drops projections outside a sensible length range
before resize-to-fixed-length.
"""

from __future__ import annotations

import polars as pl


_SCHEMA_COLS: list[str] = [
    "query_name",
    "species",
    "t_chrom",
    "t_start",
    "t_end",
    "t_strand",
    "t_src_size",
]


def filter_single_chrom_strand(records: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per ``(query_name, species)`` keeping single-chrom/strand groups.

    Output has one row per kept group with ``t_start = min(t_start)``,
    ``t_end = max(t_end)``, and the unique chrom/strand. Groups whose
    hits span multiple chroms or both strands are dropped.

    Required input columns: ``query_name``, ``species``, ``t_chrom``,
    ``t_start``, ``t_end``, ``t_strand``, ``t_src_size``.
    """
    if records.is_empty():
        return records.select(_SCHEMA_COLS)
    grouped = records.group_by(["query_name", "species"]).agg(
        n_chroms=pl.col("t_chrom").n_unique(),
        n_strands=pl.col("t_strand").n_unique(),
        t_chrom=pl.col("t_chrom").first(),
        t_strand=pl.col("t_strand").first(),
        t_start=pl.col("t_start").min(),
        t_end=pl.col("t_end").max(),
        t_src_size=pl.col("t_src_size").first(),
    )
    return grouped.filter(
        (pl.col("n_chroms") == 1) & (pl.col("n_strands") == 1)
    ).select(_SCHEMA_COLS)


def filter_length(
    records: pl.DataFrame, *, min_len: int = 128, max_len: int = 512
) -> pl.DataFrame:
    """Keep records where ``t_end - t_start`` is in ``[min_len, max_len]`` inclusive."""
    assert min_len > 0 and max_len >= min_len
    length = pl.col("t_end") - pl.col("t_start")
    return records.filter((length >= min_len) & (length <= max_len))
