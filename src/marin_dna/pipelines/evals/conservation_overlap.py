"""Overlap of eval variants with zoonomia conservation regions.

Descriptive coverage analysis for issue #213: for the matched-pair eval
datasets (``mendelian_traits`` / ``complex_traits``), how many variants fall
inside the human regions used to build the ``zoonomia_projection_dataset``?

Two granularities:

- **Window-level** (:func:`add_window_overlap`) — does the variant's base lie
  inside any kept conservation *window* (a 255 bp anchor with
  ``proportion_conserved >= cutoff``)? This is the zoonomia training
  footprint.
- **Base-level** (:func:`add_base_conservation`) — does the variant's *own
  nucleotide* clear the per-base ``phyloP_447m`` threshold?

Coordinate convention: variant ``pos`` is **1-based** (VCF-style), so the
variant base is the 0-based half-open interval ``[pos - 1, pos)``. Windows
and the bigWig are 0-based half-open. The 1-based -> 0-based conversion
happens here, once, at the variant boundary; everything downstream is
0-based per the repo convention.
"""

from collections.abc import Sequence
from pathlib import Path

import bioframe as bf
import numpy as np
import polars as pl

from marin_dna.data.intervals import GenomicSet
from marin_dna.pipelines.conservation.scoring import score_positions


def add_window_overlap(
    variants: pl.DataFrame,
    regions: pl.DataFrame,
    *,
    flag_col: str = "in_conserved_window",
) -> pl.DataFrame:
    """Flag variants whose base ``[pos - 1, pos)`` overlaps any region.

    Args:
        variants: frame with ``chrom`` (str) and ``pos`` (1-based int).
        regions: frame with ``chrom``/``start``/``end`` (0-based half-open).
            Overlapping rows are merged internally (via :class:`GenomicSet`),
            so the input may be the raw step-tiled windows.
        flag_col: name of the boolean output column.

    Returns:
        ``variants`` with an added boolean ``flag_col`` (``True`` iff the
        variant base overlaps any region), aligned to the input row order.
    """
    assert {"chrom", "pos"}.issubset(variants.columns), (
        f"variants needs chrom/pos; got {variants.columns}"
    )

    region_pd = GenomicSet(regions).to_pandas()  # merged, non-overlapping

    if len(region_pd) == 0:
        flags = np.zeros(variants.height, dtype=bool)
        return variants.with_columns(pl.Series(flag_col, flags, dtype=pl.Boolean))

    pts = variants.select(["chrom", "pos"]).with_row_index("__i").to_pandas()
    pts["chrom"] = pts["chrom"].astype(str)
    pts["start"] = pts["pos"].astype("int64") - 1  # 1-based pos -> 0-based base
    pts["end"] = pts["pos"].astype("int64")
    assert (pts["start"] >= 0).all(), "variant pos must be >= 1 (1-based)"

    counts = bf.count_overlaps(
        pts[["chrom", "start", "end", "__i"]], region_pd
    ).sort_values("__i")
    flags = counts["count"].to_numpy() > 0
    assert len(flags) == variants.height
    return variants.with_columns(pl.Series(flag_col, flags, dtype=pl.Boolean))


def add_base_conservation(
    variants: pl.DataFrame,
    bw_path: str | Path,
    threshold: float,
    *,
    chrom_prefix: str = "chr",
    value_col: str = "base_phylop",
    flag_col: str = "base_conserved",
) -> pl.DataFrame:
    """Annotate each variant with the per-base phyloP value at its position.

    Args:
        variants: frame with ``chrom`` (str) and ``pos`` (1-based int).
        bw_path: path to the phyloP bigWig (UCSC ``chr1``-style names).
        threshold: phyloP value at/above which the base is "conserved".
        chrom_prefix: prefix to bridge bare Ensembl names to the bigWig.
        value_col / flag_col: names of the float value and bool flag columns.

    Returns:
        ``variants`` with ``value_col`` (Float64; NaN at alignment gaps /
        chroms absent from the track) and ``flag_col`` (Boolean; NaN ->
        ``False``, matching the zoonomia NaN-as-non-conserved convention).
    """
    assert {"chrom", "pos"}.issubset(variants.columns), (
        f"variants needs chrom/pos; got {variants.columns}"
    )
    chroms = variants["chrom"].to_list()
    pos0 = [int(p) - 1 for p in variants["pos"].to_list()]  # 1-based -> 0-based
    values, conserved = score_positions(
        bw_path, chroms, pos0, threshold, chrom_prefix=chrom_prefix
    )
    return variants.with_columns(
        pl.Series(value_col, values, dtype=pl.Float64),
        pl.Series(flag_col, conserved, dtype=pl.Boolean),
    )


def overlap_summary(
    variants: pl.DataFrame,
    flag_col: str,
    *,
    by: Sequence[str] = (),
    label_col: str = "label",
) -> pl.DataFrame:
    """Fraction of variants with ``flag_col`` True, grouped by label (+ ``by``).

    Args:
        variants: annotated frame (must contain ``label_col`` and ``flag_col``).
        flag_col: boolean column to aggregate (e.g. ``in_conserved_window``).
        by: extra grouping columns (e.g. ``("consequence_group",)``); empty
            for an overall summary.
        label_col: the positive/negative label column (boolean).

    Returns:
        One row per (``label_col``, ``*by``) group with ``n`` (group size),
        ``n_in`` (count flagged True), and ``frac_in`` (the fraction).
    """
    group = [label_col, *by]
    assert set(group + [flag_col]).issubset(variants.columns), (
        f"missing columns; have {variants.columns}"
    )
    return (
        variants.group_by(group)
        .agg(
            n=pl.len(),
            n_in=pl.col(flag_col).sum(),
            frac_in=pl.col(flag_col).mean(),
        )
        .sort(group)
    )


def centered_windows(
    variants: pl.DataFrame,
    window_size: int,
    *,
    chrom_sizes: dict[str, int] | None = None,
    start_col: str = "start",
    end_col: str = "end",
) -> pl.DataFrame:
    """Build a ``window_size`` bp window centered on each variant base.

    The variant base is the 0-based ``[pos - 1, pos)`` interval (``pos`` is
    1-based). The window is ``[base - window_size // 2, base - window_size //
    2 + window_size)``; for odd ``window_size`` the variant base sits exactly
    at the center (offset ``window_size // 2``), matching the eval-harness
    ``var_pos = window_size // 2`` convention and the zoonomia 255 bp anchor
    size.

    Windows are clipped to valid chromosome bounds: ``start`` never goes
    below 0, and ``end`` is capped at the chromosome length when
    ``chrom_sizes`` is supplied (variants within ``window_size // 2`` bp of a
    telomere therefore get a slightly shorter window — ``proportion_conserved``
    stays well-defined since :func:`score_windows` divides by the actual
    ``end - start``). Row order is preserved.

    Args:
        variants: frame with ``chrom`` (str) and ``pos`` (1-based int).
        window_size: window length in bp (e.g. 255).
        chrom_sizes: optional ``{chrom: length}`` (keyed by the *variant*
            chrom names) used to cap ``end`` at the chromosome boundary.
        start_col / end_col: names of the 0-based half-open output columns.

    Returns:
        ``variants`` with added 0-based half-open ``start_col`` / ``end_col``
        columns, ready to hand to ``score_windows``.
    """
    assert window_size > 0, f"window_size must be positive, got {window_size}"
    assert {"chrom", "pos"}.issubset(variants.columns), (
        f"variants needs chrom/pos; got {variants.columns}"
    )
    half = window_size // 2
    out = variants.with_columns(
        # 0-based base = pos - 1; left edge = base - half; clip at 0.
        pl.max_horizontal(pl.col("pos") - 1 - half, pl.lit(0)).alias(start_col),
    ).with_columns(
        (pl.col(start_col) + window_size).alias(end_col),
    )
    if chrom_sizes is not None:
        missing = set(out["chrom"].unique().to_list()) - set(chrom_sizes)
        assert not missing, f"chroms absent from chrom_sizes: {sorted(missing)}"
        out = out.with_columns(
            pl.min_horizontal(
                pl.col(end_col),
                pl.col("chrom").replace_strict(chrom_sizes, return_dtype=pl.Int64),
            ).alias(end_col),
        )
    assert (out[end_col] > out[start_col]).all(), "empty/inverted window produced"
    return out
