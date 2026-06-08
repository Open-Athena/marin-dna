"""Per-position genomic annotation for the stratified-LL-gap analysis (issue #296).

**Stage 2 helpers** — pure CPU, no model. The per-token loss cache (Stage 1,
``per_token_loss.compute_hf_per_token_loss``) carries a genomic position for
every scored base; these helpers attach the annotations the analysis strata are
built from. The headline metric is the conserved/non-conserved LL gap restricted
to the **first two codon positions**, so codon-position assignment is the
correctness-critical piece and gets the most care here.

Codon position is assigned **frame-free**: a transcript's CDS segments are
concatenated in transcription order and the coding bases numbered ``0..L-1``,
with ``codon_pos = ordinal % 3 + 1``. Translation starts at a codon boundary, so
this needs no GTF ``frame`` field — but when ``frame`` is present it is used as an
independent cross-check (``frame == (-offset) % 3`` per segment), turning a
classic off-by-one risk into a loud failure (CLAUDE.md: fail fast on
silent-corruption risks).
"""

from __future__ import annotations

import polars as pl

_CDS_REQUIRED = {"chrom", "start", "end", "strand", "transcript_id"}


def cds_codon_positions(cds: pl.DataFrame) -> pl.DataFrame:
    """Expand canonical CDS segments → one row per coding base with its codon
    position (1/2/3), via frame-free transcription-order numbering.

    Args:
        cds: CDS segments, columns ``[chrom, start, end, strand, transcript_id]``
            (0-based half-open; one row per CDS exon piece). Restrict to canonical
            transcripts upstream (``validation.filter_to_canonical_transcripts``)
            so each gene contributes one reading frame. If a ``frame`` column
            (GTF phase, 0/1/2) is present it is cross-checked against the derived
            numbering and a mismatch raises.

    Returns:
        ``[chrom, genomic_pos, strand, transcript_id, ordinal, codon_pos]`` — one
        row per coding base of every input transcript. ``ordinal`` is the 0-based
        index of the base within its transcript's CDS (transcription order);
        ``codon_pos`` ∈ {1, 2, 3}. A genomic position covered by >1 transcript's
        CDS appears once per transcript (caller resolves any conflict).

    The reading frame is carried across splice junctions: the last partial codon
    of one CDS segment continues into the next, because ``ordinal`` runs over the
    concatenated CDS, not per-segment.
    """
    missing = _CDS_REQUIRED - set(cds.columns)
    assert not missing, f"cds missing columns: {missing}"
    assert len(cds) > 0, "empty cds frame"
    assert cds["strand"].is_in(["+", "-"]).all(), "strand must be + or -"
    assert (cds["end"] > cds["start"]).all(), "non-positive CDS segment"

    # Order segments within each transcript in transcription order (+ ascending
    # start, - descending start) via a signed sort key, then the cumulative CDS
    # length *before* each segment is its offset into the concatenated CDS.
    ordered = cds.with_columns(
        length=(pl.col("end") - pl.col("start")),
        _ord_key=pl.when(pl.col("strand") == "+")
        .then(pl.col("start"))
        .otherwise(-pl.col("start")),
    ).sort(["transcript_id", "_ord_key"])
    ordered = ordered.with_columns(
        offset=(pl.col("length").cum_sum().over("transcript_id") - pl.col("length"))
    )

    if "frame" in cds.columns:
        _assert_frame_consistent(ordered)

    # Expand to per-base rows; ordinal = offset + within-segment index, where the
    # within-segment index runs 5'→3' in the transcript's reading direction.
    per_base = (
        ordered.with_columns(genomic_pos=pl.int_ranges(pl.col("start"), pl.col("end")))
        .explode("genomic_pos")
        .with_columns(
            ordinal=pl.when(pl.col("strand") == "+")
            .then(pl.col("offset") + (pl.col("genomic_pos") - pl.col("start")))
            .otherwise(pl.col("offset") + (pl.col("end") - 1 - pl.col("genomic_pos")))
        )
        .with_columns(codon_pos=(pl.col("ordinal") % 3 + 1).cast(pl.Int8))
        .select(
            ["chrom", "genomic_pos", "strand", "transcript_id", "ordinal", "codon_pos"]
        )
    )
    return per_base


def _assert_frame_consistent(ordered: pl.DataFrame) -> None:
    """Cross-check the derived per-segment ``offset`` against the GTF ``frame``.

    GTF phase of a CDS segment is the number of bases to skip from its 5' end to
    the first complete codon, which equals ``(-offset) % 3`` for the frame-free
    numbering. Only rows whose ``frame`` is a valid phase (0/1/2) are checked —
    ``.`` / null phases (rare, non-CDS rows) are skipped.
    """
    check = ordered.with_columns(
        _frame_int=pl.col("frame").cast(pl.Int64, strict=False),
        _expected=((-pl.col("offset")) % 3),
    ).filter(pl.col("_frame_int").is_in([0, 1, 2]))
    if len(check) == 0:
        return
    bad = check.filter(pl.col("_frame_int") != pl.col("_expected"))
    assert len(bad) == 0, (
        f"codon-frame cross-check failed on {len(bad)} CDS segment(s): GTF frame "
        f"≠ (-offset) % 3. First: "
        f"{bad.select(['chrom', 'start', 'end', 'strand', 'transcript_id', 'frame', 'offset']).head(3).to_dicts()}"
    )


def assign_codon_positions(cds: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """Left-join query genomic positions to their codon position.

    Args:
        cds: CDS segments (see :func:`cds_codon_positions`).
        positions: query positions, columns ``[chrom, genomic_pos]`` (0-based).

    Returns:
        ``positions`` with ``[strand, transcript_id, ordinal, codon_pos]`` added.
        Non-coding positions get nulls. A position in multiple canonical CDS
        produces multiple rows; ``codon_pos`` is null outside any CDS.
    """
    assert {"chrom", "genomic_pos"} <= set(positions.columns), (
        f"positions missing chrom/genomic_pos; got {list(positions.columns)}"
    )
    per_base = cds_codon_positions(cds)
    return positions.join(per_base, on=["chrom", "genomic_pos"], how="left")
