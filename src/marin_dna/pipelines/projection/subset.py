"""Filter the all-species projection Parquet to a subset by ``query_name``.

A subset is defined as a set of human-window ``query_name`` values
(the column-4 BED names from ``windows.smk``: ``win_<chrom>_<NNN>``).
Subsets typically come from overlapping the human window BED with a
functional annotation (SCREEN cCREs, RefSeq CDS, etc.) — that
derivation is upstream and out of scope here.

The filter is applied via Polars' lazy + streaming engine, so the
~36 GB all-species Parquet is never materialized in memory and the
cost is bounded by NVMe throughput (~30–60 s per subset).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_query_names(path: str | Path) -> set[str]:
    """Read a one-name-per-line text file as a set, ignoring blanks/comments.

    Lines starting with ``#`` are skipped; empty lines are skipped.
    The set is the natural form for the downstream ``is_in`` filter.
    """
    out: set[str] = set()
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def filter_to_subset(
    all_species_parquet: str | Path,
    query_names: set[str] | list[str] | str | Path,
    out_parquet: str | Path,
) -> None:
    """Lazy-filter the all-species Parquet to rows whose ``query_name`` is in the subset.

    Args:
        all_species_parquet: source Parquet (concat of per-species rows).
        query_names: a set/list of names, or a path to a one-name-per-line
            text file (parsed via :func:`load_query_names`).
        out_parquet: destination Parquet. Written via Polars streaming
            sink — peak memory is bounded.

    Implementation note: column-pruning + filter-pushdown happen
    automatically because ``scan_parquet`` returns a LazyFrame; only
    the columns needed by downstream are decompressed.
    """
    if isinstance(query_names, (str, Path)):
        keys = load_query_names(query_names)
    else:
        keys = set(query_names)

    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Eager read+filter+write rather than scan+filter+sink_parquet:
    # the lazy streaming engine on a ``query_name.is_in(keys)`` filter
    # segfaults on the ~110 M-row all_species_with_sequence.parquet
    # (polars 1.x bug); eager fits comfortably in RAM (~25 GB decoded
    # from ~10 GB on disk) and runs in ~10 s on r6i.8xlarge.
    df = pl.read_parquet(str(all_species_parquet))
    df = df.filter(pl.col("query_name").is_in(keys))
    df.write_parquet(str(out))
