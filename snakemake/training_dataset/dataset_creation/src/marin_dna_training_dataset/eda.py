"""
EDA utility functions for analyzing causal variants in 3' UTR and ncRNA regions.

These functions support exploratory data analysis of variant distributions
relative to genomic annotations.
"""

import polars as pl

from marin_dna.data.utils import get_mrna_exons


def extract_3_prime_utr_annotations(ann: pl.DataFrame) -> pl.DataFrame:
    """
    Extract 3' UTR annotations with transcript and gene info.

    Computes 3' UTR regions from mRNA exons and CDS boundaries. For each
    transcript, the 3' UTR is the exonic region after (+ strand) or before
    (- strand) the CDS.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with columns:
        - chrom, start, end, strand: genomic coordinates (0-based)
        - transcript_id: transcript identifier
        - cds_start, cds_end: CDS boundaries for distance calculations
        - gene_id, gene_name: gene identifiers
    """
    mrna_exons = get_mrna_exons(ann)

    # Extract CDS info
    cds = ann.filter(pl.col("feature") == "CDS").with_columns(
        pl.col("attribute")
        .str.extract(r'transcript_id "(.*?)"')
        .alias("transcript_id"),
    )

    # Get CDS boundaries per transcript
    cds_bounds = cds.group_by("transcript_id").agg(
        pl.col("start").min().alias("cds_start"),
        pl.col("end").max().alias("cds_end"),
        pl.col("strand").first(),
    )

    # Join exons with CDS boundaries
    exons_with_cds = mrna_exons.join(cds_bounds, on="transcript_id", how="inner")

    # Compute 3' UTR portions per exon
    # + strand: exon region after CDS end
    # - strand: exon region before CDS start (genomically)
    utr_exons = (
        exons_with_cds.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.max_horizontal("start", "cds_end"))
            .otherwise(pl.col("start"))
            .alias("utr_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("end"))
            .otherwise(pl.min_horizontal("end", "cds_start"))
            .alias("utr_end"),
        )
        .filter(pl.col("utr_end") > pl.col("utr_start"))
        .select(
            pl.col("chrom"),
            pl.col("utr_start").alias("start"),
            pl.col("utr_end").alias("end"),
            pl.col("strand"),
            pl.col("transcript_id"),
            pl.col("cds_start"),
            pl.col("cds_end"),
        )
    )

    # Extract gene_id from annotation for each transcript
    tx_gene = (
        ann.filter(pl.col("feature") == "transcript")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
            pl.col("attribute").str.extract(r'gene_id "(.*?)"').alias("gene_id"),
            pl.col("attribute").str.extract(r'gene "(.*?)"').alias("gene_name"),
        )
        .select(["transcript_id", "gene_id", "gene_name"])
    )

    # Join UTR regions with gene info
    return utr_exons.join(tx_gene, on="transcript_id", how="left")


def extract_cds_annotations(ann: pl.DataFrame) -> pl.DataFrame:
    """
    Extract CDS annotations for distance calculations.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with columns:
        - chrom, start, end, strand: genomic coordinates (0-based)
        - transcript_id, gene_id: identifiers
    """
    return (
        ann.filter(pl.col("feature") == "CDS")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
            pl.col("attribute").str.extract(r'gene_id "(.*?)"').alias("gene_id"),
        )
        .select(["chrom", "start", "end", "strand", "transcript_id", "gene_id"])
    )


def extract_mrna_exon_annotations(ann: pl.DataFrame) -> pl.DataFrame:
    """
    Extract mRNA exon annotations for distance calculations.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with columns:
        - chrom, start, end, strand: genomic coordinates (0-based)
        - transcript_id, gene_id: identifiers
    """
    # Get mRNA exons with gene_id from annotation
    mrna_exons = (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "(.*?)"')
            .alias("transcript_biotype"),
            pl.col("attribute").str.extract(r'gbkey "(.*?)"').alias("gbkey"),
            pl.col("attribute").str.extract(r'gene_id "(.*?)"').alias("gene_id"),
        )
        .filter((pl.col("transcript_biotype") == "mRNA") | (pl.col("gbkey") == "mRNA"))
        .select(["chrom", "start", "end", "strand", "transcript_id", "gene_id"])
    )

    return mrna_exons


def compute_genomic_distance_to_cds(
    variant_start: int,
    cds_end: int,
    cds_start: int,
    strand: str,
) -> int:
    """
    Compute genomic distance from a variant to the CDS boundary.

    This is the direct distance in genomic coordinates, including any
    introns between the variant and CDS.

    Args:
        variant_start: 0-based start position of the variant.
        cds_end: End position of CDS (for + strand).
        cds_start: Start position of CDS (for - strand).
        strand: "+" or "-".

    Returns:
        Distance in base pairs from variant to CDS.
    """
    if strand == "+":
        return variant_start - cds_end
    else:
        return cds_start - variant_start - 1


def compute_mrna_distance_to_cds(
    variant_pos: int,
    utr_exons: pl.DataFrame,
    strand: str,
) -> int:
    """
    Compute mRNA distance from a variant to the CDS boundary.

    This counts only exonic bases, excluding introns. The result is the
    distance in the mature mRNA transcript.

    Args:
        variant_pos: 0-based position of the variant.
        utr_exons: DataFrame with UTR exons for the transcript, containing
                   'start' and 'end' columns. Must be sorted appropriately
                   (by start for + strand, by end descending for - strand).
        strand: "+" or "-".

    Returns:
        Distance in mRNA bases from variant to CDS.

    Example:
        For a + strand transcript with UTR exons at [100-150, 200-300]
        and a variant at position 220:
        - First exon contributes 50 bp (150-100)
        - Position in second exon: 220-200 = 20 bp
        - Total mRNA distance: 50 + 20 = 70 bp

        The genomic distance would be 220-100 = 120 bp (includes 50 bp intron).
    """
    mrna_dist = 0

    if strand == "+":
        for row in utr_exons.iter_rows(named=True):
            exon_start, exon_end = row["start"], row["end"]
            if variant_pos >= exon_end:
                # Variant is after this exon, add full exon length
                mrna_dist += exon_end - exon_start
            elif variant_pos >= exon_start:
                # Variant is in this exon
                mrna_dist += variant_pos - exon_start
                break
    else:
        for row in utr_exons.iter_rows(named=True):
            exon_start, exon_end = row["start"], row["end"]
            if variant_pos < exon_start:
                # Variant is after this exon (in 3' direction), add full length
                mrna_dist += exon_end - exon_start
            elif variant_pos < exon_end:
                # Variant is in this exon
                mrna_dist += exon_end - variant_pos - 1
                break

    return mrna_dist


def compute_mrna_distances_for_variants(
    variants: pl.DataFrame,
    utr_annotations: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute mRNA distances for a set of 3' UTR variants.

    For each variant-transcript pair, computes both genomic and mRNA
    distance to the CDS boundary.

    Args:
        variants: DataFrame with variant-transcript overlaps, containing:
                  - chrom, pos: variant position (pos is 1-based VCF style)
                  - start: 0-based start position
                  - strand, transcript_id: from UTR annotation
                  - distance_to_cds: pre-computed genomic distance
        utr_annotations: DataFrame with all UTR exons, containing:
                         - chrom, start, end, strand, transcript_id

    Returns:
        DataFrame with columns:
        - chrom, pos: variant position
        - mrna_distance_min/max: min/max mRNA distance across transcript isoforms
        - genomic_distance_min/max: min/max genomic distance across isoforms

        Results are aggregated by (chrom, pos), showing both extremes to
        capture isoform diversity. The difference between genomic_max and
        mrna_max reveals intron contributions in multi-exon UTR isoforms.
    """
    results = []

    for row in variants.iter_rows(named=True):
        tx_id = row["transcript_id"]
        var_pos = row["start"]  # 0-based variant position
        strand = row["strand"]

        # Get all UTR exons for this transcript
        tx_utrs = utr_annotations.filter(pl.col("transcript_id") == tx_id)

        # Sort appropriately for the strand
        if strand == "+":
            tx_utrs = tx_utrs.sort("start")
        else:
            tx_utrs = tx_utrs.sort("end", descending=True)

        mrna_dist = compute_mrna_distance_to_cds(var_pos, tx_utrs, strand)

        results.append(
            {
                "chrom": row["chrom"],
                "pos": row["pos"],
                "transcript_id": tx_id,
                "mrna_distance_to_cds": mrna_dist,
                "genomic_distance_to_cds": row["distance_to_cds"],
            }
        )

    result_df = pl.DataFrame(results)

    # Aggregate by position: compute min and max distances across transcripts
    # This captures both the "closest to CDS" and "furthest from CDS" isoforms
    return result_df.group_by(["chrom", "pos"]).agg(
        [
            pl.col("mrna_distance_to_cds").min().alias("mrna_distance_min"),
            pl.col("mrna_distance_to_cds").max().alias("mrna_distance_max"),
            pl.col("genomic_distance_to_cds").min().alias("genomic_distance_min"),
            pl.col("genomic_distance_to_cds").max().alias("genomic_distance_max"),
        ]
    )
