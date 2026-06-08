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
        ``[chrom, genomic_pos, strand, transcript_id, codon_pos, codon_id]`` — one
        row per coding base. ``codon_pos`` ∈ {1, 2, 3}; ``codon_id`` groups the
        three bases of one codon within a transcript (a new id at each
        ``codon_pos == 1`` in transcription order), so callers can reconstruct the
        codon (e.g. for 4-fold-degeneracy). A genomic position covered by >1
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
        # codon_id groups the three bases of each codon: in transcription order
        # (reading key = +genomic_pos on +, −genomic_pos on −), a new codon starts
        # at every codon_pos == 1.
        .with_columns(
            _rk=pl.when(pl.col("strand") == "+")
            .then(pl.col("genomic_pos"))
            .otherwise(-pl.col("genomic_pos"))
        )
        .sort(["transcript_id", "_rk"])
        .with_columns(
            codon_id=(pl.col("codon_pos") == 1)
            .cum_sum()
            .over("transcript_id")
            .cast(pl.Int32)
        )
        .select(
            ["chrom", "genomic_pos", "strand", "transcript_id", "codon_pos", "codon_id"]
        )
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
    """Intronic splice-site regions, labelled **donor** vs **acceptor**.

    Introns are the gaps between consecutive exons of a transcript; the
    splice-site region is the ``flank`` bp reaching into the intron from each
    flanking exon. The **donor** (5′ splice site, ``GT``) is the intron end
    adjacent to the upstream exon *in transcription order*; the **acceptor**
    (3′ splice site, ``AG``) is the downstream end — so the labelling is
    strand-aware (reversed on ``−``). Short introns (< 2·``flank``) collapse to
    the whole intron on both sides. Used to split ``val_cds``'s non-coding
    positions into ``splicing`` (either side) / ``splice_donor`` / ``splice_acceptor``
    vs other-noncoding. ``flank=20`` matches the set's ``add_flank(20)`` signal.

    Args:
        exons: canonical exon segments ``[chrom, start, end, strand,
            transcript_id]`` (0-based half-open). Restrict to canonical
            transcripts upstream.
        flank: bp into the intron from each exon boundary.

    Returns:
        ``[chrom, start, end, side, strand]`` 0-based half-open intervals
        (deduplicated), ``side`` ∈ {``donor``, ``acceptor``} and ``strand`` the
        gene strand (so the FWD-reading model's approach context — CDS-primed vs
        intron-primed — can be split by strand). Empty (typed) frame if no introns.
    """
    req = {"chrom", "start", "end", "strand", "transcript_id"}
    missing = req - set(exons.columns)
    assert not missing, f"exons missing columns: {missing}"
    assert flank > 0, "flank must be positive"
    empty = pl.DataFrame(
        schema={
            "chrom": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "side": pl.Utf8,
            "strand": pl.Utf8,
        }
    )
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
            pl.col("strand"),
            pl.col("end").alias("istart"),
            pl.col("_next_start").alias("iend"),
        )
    )
    if len(introns) == 0:
        return empty

    # Low side = [istart, istart+flank); high side = [iend-flank, iend). On + the
    # low side is the donor (upstream exon is lower-coordinate); on − it flips.
    low = introns.select(
        pl.col("chrom"),
        pl.col("strand"),
        pl.col("istart").alias("start"),
        pl.min_horizontal(pl.col("istart") + flank, pl.col("iend")).alias("end"),
        _low=pl.lit(True),
    )
    high = introns.select(
        pl.col("chrom"),
        pl.col("strand"),
        pl.max_horizontal(pl.col("iend") - flank, pl.col("istart")).alias("start"),
        pl.col("iend").alias("end"),
        _low=pl.lit(False),
    )
    return (
        pl.concat([low, high])
        .with_columns(
            side=pl.when(
                ((pl.col("strand") == "+") & pl.col("_low"))
                | ((pl.col("strand") == "-") & ~pl.col("_low"))
            )
            .then(pl.lit("donor"))
            .otherwise(pl.lit("acceptor"))
        )
        .select(["chrom", "start", "end", "side", "strand"])
        .unique()
        .sort(["chrom", "start", "end", "side"])
    )


# Two-base 5' prefixes of the eight 4-fold-degenerate codon families — every
# substitution at the 3rd position is synonymous: Ser TCN, Leu CTN, Pro CCN,
# Arg CGN, Thr ACN, Val GTN, Ala GCN, Gly GGN.
FOURFOLD_PREFIXES: frozenset[str] = frozenset(
    {"TC", "CT", "CC", "CG", "AC", "GT", "GC", "GG"}
)


def flag_fourfold_degenerate(coding: pl.DataFrame) -> pl.DataFrame:
    """Mark each 3rd-codon-position base as 4-fold-degenerate or not.

    A 3rd-position base is **4-fold degenerate** when its codon's amino acid is
    fixed by the first two bases (any 3rd base is synonymous) — the sharpest
    "unconstrained coding" set, central to the #279 "model fits tolerated coding
    variation" hypothesis. Determined from the codon's first two reference bases
    (:data:`FOURFOLD_PREFIXES`).

    Args:
        coding: coding-base rows ``[chrom, genomic_pos, transcript_id, codon_id,
            codon_pos, ref_base]`` (``ref_base`` uppercased; ``codon_id`` from
            :func:`cds_codon_positions`). Codons missing base 1 or 2 (window edge,
            partial leading codon) are dropped — undeterminable.

    Returns:
        ``[chrom, genomic_pos, is_4fold]`` for the 3rd-position base of every
        fully-observed codon (one row per transcript×codon; a genomic position in
        >1 transcript may appear more than once — caller resolves).
    """
    req = {"chrom", "genomic_pos", "transcript_id", "codon_id", "codon_pos", "ref_base"}
    missing = req - set(coding.columns)
    assert not missing, f"coding missing columns: {missing}"
    by_codon = coding.group_by(["transcript_id", "codon_id"]).agg(
        b1=pl.col("ref_base").filter(pl.col("codon_pos") == 1).first(),
        b2=pl.col("ref_base").filter(pl.col("codon_pos") == 2).first(),
        p3_chrom=pl.col("chrom").filter(pl.col("codon_pos") == 3).first(),
        p3_pos=pl.col("genomic_pos").filter(pl.col("codon_pos") == 3).first(),
    )
    full = by_codon.filter(
        pl.col("b1").is_not_null()
        & pl.col("b2").is_not_null()
        & pl.col("p3_pos").is_not_null()
    )
    return full.select(
        pl.col("p3_chrom").alias("chrom"),
        pl.col("p3_pos").alias("genomic_pos"),
        is_4fold=(pl.col("b1") + pl.col("b2")).is_in(list(FOURFOLD_PREFIXES)),
    )
