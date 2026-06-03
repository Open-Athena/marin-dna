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


def load_species(path: str | Path) -> set[str]:
    """Read the ``species`` column of a species TSV as a set.

    The species TSVs (``config/species_zoonomia_447_<rank>_dedup.tsv``) are
    tab-separated with a ``species`` header column holding raw HAL leaf names
    — the same vocabulary as the ``species`` column of the projection
    Parquets, so the set is the natural form for the downstream ``is_in``
    filter.
    """
    df = pl.read_csv(str(path), separator="\t")
    assert "species" in df.columns, (
        f"species TSV {path} missing 'species' column; got {df.columns}"
    )
    out = set(df["species"].to_list())
    assert out, f"no species rows in {path}"
    return out


def filter_to_species(
    source_parquet: str | Path,
    species: set[str] | list[str] | str | Path,
    out_parquet: str | Path,
) -> None:
    """Filter a projection Parquet to rows whose ``species`` is in the cohort.

    The species cohort is a row-filter on the ``species`` column, orthogonal
    to the ``query_name`` filter applied by :func:`filter_to_subset`: the two
    compose (e.g. the v4_cds intervals subset restricted to the one-per-order
    species cohort). ``species`` is a set/list of HAL leaf names, or a path to
    a species TSV (parsed via :func:`load_species`, reading the ``species``
    column).

    Asserts the requested cohort is a **subset** of the species present in
    ``source_parquet`` — a missing leaf means the cohort references a species
    not in this projection (or one with zero rows in this intervals subset),
    which would silently shrink the dataset, so we fail loudly (CLAUDE.md
    "fail fast on silent-corruption risks").

    Eager read+filter+write rather than the lazy ``scan``/``sink_parquet``
    path — same polars-1.x large-Parquet segfault that
    :func:`filter_to_subset` documents; the source intervals subset fits
    comfortably in RAM on the upload cluster.

    Args:
        source_parquet: the intervals-subset Parquet to filter (e.g. the
            v4_cds subset, carrying all projection species' rows).
        species: a set/list of HAL leaf names, or a path to a species TSV.
        out_parquet: destination Parquet (parent dirs created as needed).
    """
    if isinstance(species, (str, Path)):
        keys = load_species(species)
    else:
        keys = set(species)
    assert keys, "empty species cohort"

    df = pl.read_parquet(str(source_parquet))
    assert "species" in df.columns, (
        f"source Parquet missing 'species' column; got {df.columns}"
    )
    present = set(df["species"].unique().to_list())
    missing = keys - present
    assert not missing, (
        f"species cohort is not a subset of the projection: "
        f"{sorted(missing)} absent from {source_parquet} "
        f"({len(present)} species present)"
    )

    out = df.filter(pl.col("species").is_in(keys))
    # Given the subset check above, every requested leaf has >=1 row, so this
    # is a guaranteed invariant — assert it anyway (loud failure beats a
    # silently short cohort feeding into training).
    kept = set(out["species"].unique().to_list())
    assert kept == keys, (
        f"expected all {len(keys)} cohort species kept; missing "
        f"{sorted(keys - kept)}"
    )
    assert out.height > 0, "species-filtered subset is empty"

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(str(out_path))
