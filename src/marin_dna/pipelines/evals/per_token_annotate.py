"""Per-position genomic annotation for the stratified-LL-gap analysis (issue #296).

**Stage 2 helpers** — pure CPU, no model. The per-token loss cache (Stage 1,
``per_token_loss.compute_hf_per_token_loss``) carries a genomic position for
every scored base; these helpers attach the annotations the analysis strata are
built from. The headline metric is the conserved/non-conserved LL gap restricted
to the **first two codon positions**, so codon-position assignment is the
correctness-critical piece and gets the most care here.

Codon position is assigned **per CDS segment from the GTF ``frame`` (phase)**:
for a coding base at 5'-reading offset ``o`` within a segment of phase ``f``,
``codon_pos = ((o − f) mod 3) + 1``. Using the phase is what makes this correct
for **partial / 5'-incomplete CDS** (TR/IG gene segments, selenoproteins, …),
whose annotated CDS starts mid-codon (``f ≠ 0``) — a "frame-free" numbering that
assumed every CDS starts on a codon boundary silently mis-assigns those. The
formula needs only each segment's own phase, so it also carries the reading
frame across splice junctions without any whole-transcript bookkeeping. The
independent off-by-one gate lives in Stage 2: the model-free conservation
signature (the 3rd/wobble position must be the least phyloP-conserved).
"""

from __future__ import annotations

import polars as pl

_CDS_REQUIRED = {"chrom", "start", "end", "strand", "transcript_id", "frame"}


def cds_codon_positions(cds: pl.DataFrame) -> pl.DataFrame:
    """Expand canonical CDS segments → one row per coding base with its codon
    position (1/2/3), from each segment's GTF ``frame`` (phase).

    For a coding base at 5'-reading offset ``o`` within a segment of phase ``f``,
    ``codon_pos = ((o − f) mod 3) + 1`` (``o = genomic_pos − start`` on ``+``;
    ``end − 1 − genomic_pos`` on ``−``). The per-segment phase is authoritative:
    it is correct for partial / 5'-incomplete CDS (``f ≠ 0``, e.g. TR/IG gene
    segments) and carries the reading frame across splice junctions with no
    whole-transcript bookkeeping.

    Args:
        cds: CDS segments, columns ``[chrom, start, end, strand, transcript_id,
            frame]`` (0-based half-open; one row per CDS exon piece; ``frame`` =
            GTF phase, 0/1/2). Restrict to canonical transcripts upstream
            (``validation.filter_to_canonical_transcripts``).

    Returns:
        ``[chrom, genomic_pos, strand, transcript_id, codon_pos]`` — one row per
        coding base. ``codon_pos`` ∈ {1, 2, 3}. A genomic position covered by >1
        transcript's CDS appears once per transcript (caller resolves conflicts).
    """
    missing = _CDS_REQUIRED - set(cds.columns)
    assert not missing, f"cds missing columns: {missing}"
    assert len(cds) > 0, "empty cds frame"
    assert cds["strand"].is_in(["+", "-"]).all(), "strand must be + or -"
    assert (cds["end"] > cds["start"]).all(), "non-positive CDS segment"

    typed = cds.with_columns(_frame=pl.col("frame").cast(pl.Int64, strict=False))
    # fill_null(False): a "." / unparseable phase casts to null, and a bare
    # .all() would *ignore* nulls and pass — force those rows to fail.
    valid = typed["_frame"].is_in([0, 1, 2]).fill_null(False)
    assert valid.all(), (
        "CDS `frame` must be GTF phase 0/1/2 for every segment; "
        f"{int((~valid).sum())} segment(s) have an invalid/missing phase"
    )

    per_base = (
        typed.with_columns(genomic_pos=pl.int_ranges(pl.col("start"), pl.col("end")))
        .explode("genomic_pos")
        .with_columns(
            # 5'-reading offset within the segment (strand-aware).
            _o=pl.when(pl.col("strand") == "+")
            .then(pl.col("genomic_pos") - pl.col("start"))
            .otherwise(pl.col("end") - 1 - pl.col("genomic_pos"))
        )
        .with_columns(
            # ((o - f) mod 3) + 1; double-mod keeps it positive regardless of the
            # backend's sign convention for a negative dividend.
            codon_pos=(((pl.col("_o") - pl.col("_frame")) % 3 + 3) % 3 + 1).cast(
                pl.Int8
            )
        )
        .select(["chrom", "genomic_pos", "strand", "transcript_id", "codon_pos"])
    )
    return per_base


def assign_codon_positions(cds: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """Left-join query genomic positions to their codon position.

    Args:
        cds: CDS segments (see :func:`cds_codon_positions`).
        positions: query positions, columns ``[chrom, genomic_pos]`` (0-based).

    Returns:
        ``positions`` with ``[strand, transcript_id, codon_pos]`` added. Non-coding
        positions get nulls. A position in multiple canonical CDS produces
        multiple rows; ``codon_pos`` is null outside any CDS.
    """
    assert {"chrom", "genomic_pos"} <= set(positions.columns), (
        f"positions missing chrom/genomic_pos; got {list(positions.columns)}"
    )
    per_base = cds_codon_positions(cds)
    return positions.join(per_base, on=["chrom", "genomic_pos"], how="left")


def intron_splice_regions(exons: pl.DataFrame, *, flank: int = 20) -> pl.DataFrame:
    """Intronic splice-site regions: the ``flank`` bp of each intron adjacent to
    an exon (the donor and acceptor sides of every splice junction).

    Introns are the gaps between consecutive exons of a transcript; the
    splice-site region is the ``flank`` bp reaching into the intron from each
    flanking exon. Short introns (< 2·``flank``) collapse to the whole intron.
    Used to split ``val_cds``'s non-coding positions into **splicing** (intron
    ≤ ``flank`` bp from a junction — donor ``GT`` / acceptor ``AG``) vs
    other-noncoding (UTR, deeper intron). ``flank=20`` matches the set's
    ``add_flank(20)`` splice signal.

    Args:
        exons: canonical exon segments with columns ``[chrom, start, end,
            transcript_id]`` (0-based half-open). One transcript's exons define
            its introns; restrict to canonical transcripts upstream.
        flank: bp into the intron from each exon boundary.

    Returns:
        ``[chrom, start, end]`` 0-based half-open intervals (deduplicated), the
        intronic splice-site regions. Empty (typed) frame if there are no introns.
    """
    req = {"chrom", "start", "end", "transcript_id"}
    missing = req - set(exons.columns)
    assert not missing, f"exons missing columns: {missing}"
    assert flank > 0, "flank must be positive"
    empty = pl.DataFrame(schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64})
    if len(exons) == 0:
        return empty

    introns = (
        exons.sort(["transcript_id", "start"])
        .with_columns(_next_start=pl.col("start").shift(-1).over("transcript_id"))
        .filter(
            pl.col("_next_start").is_not_null()
            & (pl.col("_next_start") > pl.col("end"))
        )
        .select(
            pl.col("chrom"),
            pl.col("end").alias("istart"),
            pl.col("_next_start").alias("iend"),
        )
    )
    if len(introns) == 0:
        return empty

    donor = introns.select(
        pl.col("chrom"),
        pl.col("istart").alias("start"),
        pl.min_horizontal(pl.col("istart") + flank, pl.col("iend")).alias("end"),
    )
    acceptor = introns.select(
        pl.col("chrom"),
        pl.max_horizontal(pl.col("iend") - flank, pl.col("istart")).alias("start"),
        pl.col("iend").alias("end"),
    )
    return pl.concat([donor, acceptor]).unique().sort(["chrom", "start", "end"])
