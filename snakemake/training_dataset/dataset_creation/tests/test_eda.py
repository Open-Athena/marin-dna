"""Tests for EDA utility functions."""

import polars as pl
import pytest
from marin_dna_training_dataset.eda import (
    compute_genomic_distance_to_cds,
    compute_mrna_distance_to_cds,
    extract_3_prime_utr_annotations,
    extract_cds_annotations,
    extract_mrna_exon_annotations,
)

# =============================================================================
# Fixtures
# =============================================================================
# Note: chrY_annotation fixture is defined in conftest.py


@pytest.fixture
def simple_annotation():
    """Create a simple annotation DataFrame for unit tests.

    Creates a transcript with:
    - Gene on + strand
    - CDS from 200-500
    - 3' UTR in two exons: 500-600 and 700-900 (100 bp intron)

    This allows testing both genomic and mRNA distance calculations.
    """
    return pl.DataFrame(
        {
            "chrom": [
                "chr1",
                "chr1",
                "chr1",
                "chr1",
                "chr1",  # transcript 1
                "chr1",
                "chr1",
                "chr1",
                "chr1",
                "chr1",  # transcript 2 (- strand)
            ],
            "source": ["test"] * 10,
            "feature": [
                "transcript",
                "exon",
                "exon",
                "exon",
                "CDS",
                "transcript",
                "exon",
                "exon",
                "exon",
                "CDS",
            ],
            "start": [
                100,
                100,
                500,
                700,
                200,  # tx1: exons at 100-200, 500-600, 700-900
                1000,
                1000,
                1200,
                1500,
                1200,  # tx2: - strand
            ],
            "end": [
                900,
                200,
                600,
                900,
                500,
                1900,
                1100,
                1400,
                1900,
                1800,
            ],
            "score": ["."] * 10,
            "strand": ["+", "+", "+", "+", "+", "-", "-", "-", "-", "-"],
            "frame": ["."] * 10,
            "attribute": [
                'gene_id "gene1"; transcript_id "tx1"; gene "GENE1"; transcript_biotype "mRNA"',
                'gene_id "gene1"; transcript_id "tx1"; transcript_biotype "mRNA"',
                'gene_id "gene1"; transcript_id "tx1"; transcript_biotype "mRNA"',
                'gene_id "gene1"; transcript_id "tx1"; transcript_biotype "mRNA"',
                'gene_id "gene1"; transcript_id "tx1"',
                'gene_id "gene2"; transcript_id "tx2"; gene "GENE2"; transcript_biotype "mRNA"',
                'gene_id "gene2"; transcript_id "tx2"; transcript_biotype "mRNA"',
                'gene_id "gene2"; transcript_id "tx2"; transcript_biotype "mRNA"',
                'gene_id "gene2"; transcript_id "tx2"; transcript_biotype "mRNA"',
                'gene_id "gene2"; transcript_id "tx2"',
            ],
        }
    )


# =============================================================================
# Unit Tests: compute_genomic_distance_to_cds
# =============================================================================


def test_genomic_distance_plus_strand():
    """Test genomic distance calculation for + strand.

    Variant at position 550, CDS ends at 500.
    Expected: 550 - 500 = 50 bp
    """
    result = compute_genomic_distance_to_cds(
        variant_start=550,
        cds_end=500,
        cds_start=200,
        strand="+",
    )
    assert result == 50


def test_genomic_distance_minus_strand():
    """Test genomic distance calculation for - strand.

    Variant at position 150, CDS starts at 200.
    Expected: 200 - 150 - 1 = 49 bp
    """
    result = compute_genomic_distance_to_cds(
        variant_start=150,
        cds_end=500,
        cds_start=200,
        strand="-",
    )
    assert result == 49


def test_genomic_distance_at_cds_boundary_plus():
    """Test genomic distance when variant is at CDS boundary (+ strand).

    Variant at position 500, CDS ends at 500.
    Expected: 500 - 500 = 0 bp
    """
    result = compute_genomic_distance_to_cds(
        variant_start=500,
        cds_end=500,
        cds_start=200,
        strand="+",
    )
    assert result == 0


# =============================================================================
# Unit Tests: compute_mrna_distance_to_cds
# =============================================================================


def test_mrna_distance_single_exon_plus_strand():
    """Test mRNA distance in single-exon UTR (+ strand).

    UTR exon: 500-600 (100 bp)
    Variant at position 550
    Expected: 550 - 500 = 50 bp
    """
    utr_exons = pl.DataFrame(
        {
            "start": [500],
            "end": [600],
        }
    )

    result = compute_mrna_distance_to_cds(
        variant_pos=550,
        utr_exons=utr_exons,
        strand="+",
    )
    assert result == 50


def test_mrna_distance_multi_exon_plus_strand():
    """Test mRNA distance in multi-exon UTR (+ strand).

    UTR exons: 500-600 (100 bp) and 700-900 (200 bp)
    Variant at position 750 (in second exon)

    Genomic distance: 750 - 500 = 250 bp (includes 100 bp intron)
    mRNA distance: 100 (first exon) + 50 (position in second) = 150 bp
    """
    utr_exons = pl.DataFrame(
        {
            "start": [500, 700],
            "end": [600, 900],
        }
    ).sort("start")

    result = compute_mrna_distance_to_cds(
        variant_pos=750,
        utr_exons=utr_exons,
        strand="+",
    )
    assert result == 150


def test_mrna_distance_multi_exon_minus_strand():
    """Test mRNA distance in multi-exon UTR (- strand).

    For - strand, 3' UTR is at the genomic 5' end (lower coordinates).
    UTR exons: 100-200 (100 bp) and 300-400 (100 bp)
    CDS is at higher coordinates.

    Variant at position 150 (in first exon, which is 3'-most in mRNA)
    mRNA distance from 3' end of CDS:
    - Second exon (300-400) contributes 100 bp
    - Position in first exon: 200 - 150 - 1 = 49 bp
    - Total: 100 + 49 = 149 bp
    """
    # For - strand, exons should be sorted by end descending
    utr_exons = pl.DataFrame(
        {
            "start": [300, 100],
            "end": [400, 200],
        }
    )

    result = compute_mrna_distance_to_cds(
        variant_pos=150,
        utr_exons=utr_exons,
        strand="-",
    )
    assert result == 149


def test_mrna_distance_variant_in_first_exon():
    """Test mRNA distance when variant is in first UTR exon.

    UTR exons: 500-600, 700-900
    Variant at position 520
    Expected: 520 - 500 = 20 bp (no contribution from later exons)
    """
    utr_exons = pl.DataFrame(
        {
            "start": [500, 700],
            "end": [600, 900],
        }
    ).sort("start")

    result = compute_mrna_distance_to_cds(
        variant_pos=520,
        utr_exons=utr_exons,
        strand="+",
    )
    assert result == 20


# =============================================================================
# Unit Tests: extract_3_prime_utr_annotations
# =============================================================================


def test_extract_3_prime_utr_annotations_plus_strand(simple_annotation):
    """Test 3' UTR extraction for + strand transcript.

    tx1 has CDS 200-500, exons at 500-600 and 700-900.
    3' UTR should be the two exons after CDS: 500-600 and 700-900.
    """
    result = extract_3_prime_utr_annotations(simple_annotation)

    tx1_utrs = result.filter(pl.col("transcript_id") == "tx1").sort("start")

    assert len(tx1_utrs) == 2
    assert tx1_utrs["start"].to_list() == [500, 700]
    assert tx1_utrs["end"].to_list() == [600, 900]
    assert tx1_utrs["cds_end"][0] == 500


def test_extract_3_prime_utr_annotations_minus_strand(simple_annotation):
    """Test 3' UTR extraction for - strand transcript.

    tx2 has CDS 1200-1800, exons at 1000-1100, 1200-1400, 1500-1900.
    3' UTR should be the exon before CDS start: 1000-1100.
    """
    result = extract_3_prime_utr_annotations(simple_annotation)

    tx2_utrs = result.filter(pl.col("transcript_id") == "tx2").sort("start")

    assert len(tx2_utrs) == 1
    assert tx2_utrs["start"].to_list() == [1000]
    assert tx2_utrs["end"].to_list() == [1100]
    assert tx2_utrs["cds_start"][0] == 1200


def test_extract_3_prime_utr_annotations_includes_gene_info(simple_annotation):
    """Test that gene_id and gene_name are included in output."""
    result = extract_3_prime_utr_annotations(simple_annotation)

    tx1_utrs = result.filter(pl.col("transcript_id") == "tx1")
    assert tx1_utrs["gene_id"][0] == "gene1"
    assert tx1_utrs["gene_name"][0] == "GENE1"


# =============================================================================
# Unit Tests: extract_cds_annotations
# =============================================================================


def test_extract_cds_annotations(simple_annotation):
    """Test CDS extraction."""
    result = extract_cds_annotations(simple_annotation)

    assert len(result) == 2
    assert set(result["transcript_id"].to_list()) == {"tx1", "tx2"}
    assert "gene_id" in result.columns


# =============================================================================
# Integration Tests with chrY fixture
# =============================================================================


def test_extract_3_prime_utr_annotations_chrY(chrY_annotation):
    """Integration test: extract 3' UTRs from real chrY annotation."""
    result = extract_3_prime_utr_annotations(chrY_annotation)

    # Should have some UTRs
    assert len(result) > 0

    # Check required columns
    required_cols = [
        "chrom",
        "start",
        "end",
        "strand",
        "transcript_id",
        "cds_start",
        "cds_end",
        "gene_id",
    ]
    for col in required_cols:
        assert col in result.columns

    # All should be on chrY
    assert result["chrom"].unique().to_list() == ["NC_000024.10"]

    # Start should be less than end
    assert (result["start"] < result["end"]).all()


def test_extract_cds_annotations_chrY(chrY_annotation):
    """Integration test: extract CDS from real chrY annotation."""
    result = extract_cds_annotations(chrY_annotation)

    assert len(result) > 0
    assert result["chrom"].unique().to_list() == ["NC_000024.10"]


def test_extract_mrna_exon_annotations_chrY(chrY_annotation):
    """Integration test: extract mRNA exons from real chrY annotation."""
    result = extract_mrna_exon_annotations(chrY_annotation)

    assert len(result) > 0
    assert result["chrom"].unique().to_list() == ["NC_000024.10"]


def test_mrna_vs_genomic_distance_chrY(chrY_annotation):
    """Integration test: verify mRNA distance <= genomic distance.

    mRNA distance excludes introns, so it should always be <= genomic distance.
    """
    utr_annotations = extract_3_prime_utr_annotations(chrY_annotation)

    # Find a transcript with multiple UTR exons
    multi_exon_txs = (
        utr_annotations.group_by("transcript_id").len().filter(pl.col("len") > 1)
    )

    if len(multi_exon_txs) == 0:
        pytest.skip("No multi-exon UTRs found in chrY")

    tx_id = multi_exon_txs["transcript_id"][0]
    tx_utrs = utr_annotations.filter(pl.col("transcript_id") == tx_id)
    strand = tx_utrs["strand"][0]
    cds_end = tx_utrs["cds_end"][0]
    cds_start = tx_utrs["cds_start"][0]

    # Sort appropriately
    if strand == "+":
        tx_utrs = tx_utrs.sort("start")
        # Test a position in the last exon
        last_exon = tx_utrs.tail(1)
        test_pos = (last_exon["start"][0] + last_exon["end"][0]) // 2
    else:
        tx_utrs = tx_utrs.sort("end", descending=True)
        last_exon = tx_utrs.tail(1)
        test_pos = (last_exon["start"][0] + last_exon["end"][0]) // 2

    mrna_dist = compute_mrna_distance_to_cds(test_pos, tx_utrs, strand)
    genomic_dist = compute_genomic_distance_to_cds(test_pos, cds_end, cds_start, strand)

    # mRNA distance should be <= genomic distance (equality if single exon or no introns)
    assert mrna_dist <= genomic_dist, (
        f"mRNA distance ({mrna_dist}) > genomic distance ({genomic_dist}) "
        f"for transcript {tx_id}"
    )
