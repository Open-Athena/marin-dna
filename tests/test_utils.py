import numpy as np
import pandas as pd
import polars as pl
from Bio.Seq import Seq

from marin_dna.data.intervals import GenomicSet
from marin_dna.data.utils import (
    add_rc,
    get_3_prime_utr,
    get_5_prime_utr,
    get_array_split_pairs,
    get_cds,
    get_downstream_of_CDS,
    get_exons,
    get_ensembl_functional_exons,
    get_exons_for_masking,
    get_mrna_exons,
    get_ncrna_exons,
    get_promoters,
    get_promoters_from_exons,
    get_upstream_of_CDS,
    load_annotation,
    load_fasta,
    read_bed_to_pandas,
    write_pandas_to_bed,
)


def test_get_array_split_pairs_even_division():
    """Test get_array_split_pairs with evenly divisible length.

    Input: L=10, n_shards=2
    Output: [(0, 5), (5, 10)] (each shard has size 5)
    """
    result = get_array_split_pairs(L=10, n_shards=2)
    expected = [(0, 5), (5, 10)]
    assert result == expected


def test_get_array_split_pairs_uneven_division():
    """Test get_array_split_pairs with remainder.

    Input: L=10, n_shards=3
    Output: [(0, 4), (4, 7), (7, 10)] (first shard gets extra element)

    Note: With L=10 and n_shards=3, base_size=3, remainder=1.
    First shard gets size 4, remaining shards get size 3.
    """
    result = get_array_split_pairs(L=10, n_shards=3)
    expected = [(0, 4), (4, 7), (7, 10)]
    assert result == expected


def test_get_array_split_pairs_matches_numpy():
    """Test that get_array_split_pairs matches np.array_split behavior.

    Input: L=23, n_shards=5
    Output: Slices matching np.array_split result
    """
    L = 23
    n_shards = 5
    result = get_array_split_pairs(L, n_shards)

    # Create a dummy array and split it with numpy
    arr = np.arange(L)
    np_splits = np.array_split(arr, n_shards)

    # Verify each slice matches numpy's split
    for i, (start, end) in enumerate(result):
        np.testing.assert_array_equal(arr[start:end], np_splits[i])


def test_get_array_split_pairs_single_shard():
    """Test get_array_split_pairs with n_shards=1.

    Input: L=100, n_shards=1
    Output: [(0, 100)] (single shard contains all elements)
    """
    result = get_array_split_pairs(L=100, n_shards=1)
    expected = [(0, 100)]
    assert result == expected


def test_get_array_split_pairs_many_shards():
    """Test get_array_split_pairs with n_shards equal to length.

    Input: L=5, n_shards=5
    Output: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)] (each shard has size 1)
    """
    result = get_array_split_pairs(L=5, n_shards=5)
    expected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert result == expected


def test_get_array_split_pairs_more_shards_than_length():
    """Test get_array_split_pairs with n_shards > L.

    Input: L=3, n_shards=5
    Output: [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)] (last shards are empty)

    Note: When n_shards > L, some shards will be empty (start == end).
    This matches np.array_split behavior.
    """
    result = get_array_split_pairs(L=3, n_shards=5)
    expected = [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)]
    assert result == expected


def test_get_array_split_pairs_empty_array():
    """Test get_array_split_pairs with empty array (L=0).

    Input: L=0, n_shards=3
    Output: [(0, 0), (0, 0), (0, 0)] (all shards are empty)
    """
    result = get_array_split_pairs(L=0, n_shards=3)
    expected = [(0, 0), (0, 0), (0, 0)]
    assert result == expected


def test_get_array_split_pairs_covers_full_range():
    """Test that slices cover the full range without gaps or overlaps.

    Input: L=17, n_shards=4
    Output: Slices that cover [0, 17) exactly once
    """
    L = 17
    n_shards = 4
    result = get_array_split_pairs(L, n_shards)

    # First slice should start at 0
    assert result[0][0] == 0
    # Last slice should end at L
    assert result[-1][1] == L

    # Each slice end should equal next slice start (no gaps or overlaps)
    for i in range(len(result) - 1):
        assert result[i][1] == result[i + 1][0]


def test_get_array_split_pairs_correct_number_of_shards():
    """Test that the number of returned slices equals n_shards.

    Input: L=100, n_shards=7
    Output: List with exactly 7 tuples
    """
    result = get_array_split_pairs(L=100, n_shards=7)
    assert len(result) == 7


def test_get_array_split_pairs_large_values():
    """Test get_array_split_pairs with large values.

    Input: L=1000000, n_shards=13
    Output: 13 slices covering the full range
    """
    L = 1000000
    n_shards = 13
    result = get_array_split_pairs(L, n_shards)

    assert len(result) == n_shards
    assert result[0][0] == 0
    assert result[-1][1] == L

    # Verify total size
    total_size = sum(end - start for start, end in result)
    assert total_size == L


def test_get_array_split_pairs_remainder_distribution():
    """Test that remainder is distributed to first shards.

    Input: L=23, n_shards=5
    Output: First 3 shards get size 5, last 2 get size 4

    Note: base_size=4, remainder=3, so first 3 shards get size 5.
    """
    result = get_array_split_pairs(L=23, n_shards=5)
    sizes = [end - start for start, end in result]

    # First 3 shards should have size 5 (base_size + 1)
    # Last 2 shards should have size 4 (base_size)
    assert sizes == [5, 5, 5, 4, 4]


def test_get_array_split_pairs_all_equal_when_divisible():
    """Test that all shards have equal size when L is divisible by n_shards.

    Input: L=20, n_shards=4
    Output: All shards have size 5
    """
    result = get_array_split_pairs(L=20, n_shards=4)
    sizes = [end - start for start, end in result]

    # All shards should have equal size when evenly divisible
    assert all(size == 5 for size in sizes)


def test_get_array_split_pairs_consistency_with_numpy_various_inputs():
    """Test consistency with np.array_split across various inputs.

    Tests multiple combinations of L and n_shards to ensure consistent behavior.
    """
    test_cases = [
        (10, 3),
        (100, 7),
        (7, 10),
        (50, 50),
        (1, 1),
        (0, 5),
        (15, 4),
    ]

    for L, n_shards in test_cases:
        result = get_array_split_pairs(L, n_shards)
        arr = np.arange(L)
        np_splits = np.array_split(arr, n_shards)

        # Verify each slice matches numpy's split
        for i, (start, end) in enumerate(result):
            np.testing.assert_array_equal(
                arr[start:end],
                np_splits[i],
                err_msg=f"Mismatch for L={L}, n_shards={n_shards}, shard {i}",
            )


# load_annotation tests
def test_load_annotation_basic(tmp_path):
    """Test load_annotation with basic GTF file.

    Input: GTF file with 2 features on '+' and '-' strands
    Output: DataFrame with start coordinates converted to 0-based (BED)
    """
    gtf_file = tmp_path / "test.gtf"
    gtf_content = """chr1\ttest\tgene\t1000\t2000\t.\t+\t.\tgene_id "gene1"
chr1\ttest\texon\t1500\t1800\t.\t-\t.\tgene_id "gene1"
chr2\ttest\tCDS\t500\t600\t.\t+\t.\tgene_id "gene2"
"""
    gtf_file.write_text(gtf_content)

    result = load_annotation(str(gtf_file))

    # Check start coordinates are converted from 1-based to 0-based
    assert result["start"].to_list() == [999, 1499, 499]
    assert result["end"].to_list() == [2000, 1800, 600]
    assert result["strand"].to_list() == ["+", "-", "+"]
    assert len(result) == 3


def test_load_annotation_filters_invalid_strands(tmp_path):
    """Test that load_annotation filters out features without valid strand.

    Input: GTF with '.' strand (invalid)
    Output: Empty DataFrame (invalid strand filtered out)
    """
    gtf_file = tmp_path / "test.gtf"
    gtf_content = """chr1\ttest\tgene\t1000\t2000\t.\t.\t.\tgene_id "gene1"
"""
    gtf_file.write_text(gtf_content)

    result = load_annotation(str(gtf_file))

    # Invalid strand should be filtered out
    assert len(result) == 0


def test_load_annotation_handles_comments(tmp_path):
    """Test that load_annotation skips comment lines.

    Input: GTF file with comment lines starting with '#'
    Output: DataFrame without comment lines
    """
    gtf_file = tmp_path / "test.gtf"
    gtf_content = """# This is a comment
#Another comment
chr1\ttest\tgene\t1000\t2000\t.\t+\t.\tgene_id "gene1"
"""
    gtf_file.write_text(gtf_content)

    result = load_annotation(str(gtf_file))

    assert len(result) == 1
    assert result["chrom"].to_list() == ["chr1"]


# get_mrna_exons tests
def test_get_mrna_exons_with_transcript_biotype():
    """Test get_mrna_exons with transcript_biotype annotation.

    Input: Annotation with mRNA exons (using transcript_biotype)
    Output: DataFrame with only mRNA exons
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [150, 250, 350],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans2"; transcript_biotype "lncRNA"',
                'transcript_id "trans3"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_mrna_exons(ann)

    # Only the first exon should be retained (mRNA exon)
    assert len(result) == 1
    assert result["transcript_id"].to_list() == ["trans1"]
    assert result["start"].to_list() == [100]


def test_get_mrna_exons_with_gbkey():
    """Test get_mrna_exons with gbkey annotation.

    Input: Annotation with mRNA exons (using gbkey)
    Output: DataFrame with only mRNA exons
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [150, 250],
            "strand": ["+", "-"],
            "feature": ["exon", "exon"],
            "attribute": [
                'transcript_id "trans1"; gbkey "mRNA"',
                'transcript_id "trans2"; gbkey "tRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_mrna_exons(ann)

    # Only the first exon should be retained (mRNA via gbkey)
    assert len(result) == 1
    assert result["transcript_id"].to_list() == ["trans1"]
    assert result["strand"].to_list() == ["+"]


def test_get_mrna_exons_filters_non_exons():
    """Test that get_mrna_exons filters out non-exon features.

    Input: Annotation with CDS and gene features (no exons)
    Output: Empty DataFrame
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [150, 250],
            "strand": ["+", "+"],
            "feature": ["CDS", "gene"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'gene_id "gene1"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_mrna_exons(ann)

    assert len(result) == 0


# get_promoters_from_exons tests
def test_get_promoters_from_exons_positive_strand():
    """Test get_promoters_from_exons for genes on positive strand.

    Input: Exons for gene on '+' strand, n_upstream=100, n_downstream=50
    Output: Promoter region [TSS-100, TSS+50]
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1500],
            "end": [1200, 1700],
            "strand": ["+", "+"],
            "transcript_id": ["trans1", "trans1"],
        }
    )

    result = get_promoters_from_exons(exons, n_upstream=100, n_downstream=50)

    # For '+' strand, TSS is at min(start) = 1000
    # Promoter is [1000-100, 1000+50] = [900, 1050]
    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [900]
    assert df["end"].to_list() == [1050]


def test_get_promoters_from_exons_negative_strand():
    """Test get_promoters_from_exons for genes on negative strand.

    Input: Exons for gene on '-' strand, n_upstream=100, n_downstream=50
    Output: Promoter region [TSS-50, TSS+100]
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1500],
            "end": [1200, 1700],
            "strand": ["-", "-"],
            "transcript_id": ["trans1", "trans1"],
        }
    )

    result = get_promoters_from_exons(exons, n_upstream=100, n_downstream=50)

    # For '-' strand, TSS is at max(end) = 1700
    # Promoter is [1700-50, 1700+100] = [1650, 1800]
    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [1650]
    assert df["end"].to_list() == [1800]


def test_get_promoters_from_exons_multiple_transcripts():
    """Test get_promoters_from_exons with multiple transcripts.

    Input: Exons for 2 different transcripts
    Output: 2 unique promoter regions
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1000, 1500, 2000, 2500],
            "end": [1200, 1700, 2200, 2700],
            "strand": ["+", "+", "-", "-"],
            "transcript_id": ["trans1", "trans1", "trans2", "trans2"],
        }
    )

    result = get_promoters_from_exons(exons, n_upstream=100, n_downstream=50)

    # Should have 2 promoter regions
    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 2


def test_get_promoters_from_exons_merges_overlapping():
    """Test that get_promoters_from_exons merges overlapping regions.

    Input: Two transcripts with overlapping promoter regions
    Output: Single merged promoter region
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1010],
            "end": [1200, 1210],
            "strand": ["+", "+"],
            "transcript_id": ["trans1", "trans2"],
        }
    )

    result = get_promoters_from_exons(exons, n_upstream=100, n_downstream=50)

    # GenomicSet merges overlapping intervals
    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1


# get_cds tests
def test_get_cds_returns_genomic_set():
    """Test that get_cds returns a GenomicSet.

    Input: Annotation with CDS features
    Output: GenomicSet with merged CDS regions
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [100, 200, 300],
            "end": [150, 250, 350],
            "strand": ["+", "+", "-"],
            "feature": ["CDS", "CDS", "CDS"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
                'transcript_id "trans3"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_cds(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 3


def test_get_cds_filters_non_cds():
    """Test that get_cds filters out non-CDS features.

    Input: Annotation with exon, gene, and CDS features
    Output: GenomicSet with only CDS features
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [150, 250, 350],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "CDS", "gene"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
                'gene_id "gene1"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_cds(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [200]
    assert df["end"].to_list() == [250]


def test_get_cds_merges_overlapping():
    """Test that get_cds merges overlapping CDS regions.

    Input: Annotation with overlapping CDS regions
    Output: GenomicSet with merged regions
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 120],
            "end": [150, 170],
            "strand": ["+", "+"],
            "feature": ["CDS", "CDS"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_cds(ann)

    assert isinstance(result, GenomicSet)
    # Overlapping [100-150] and [120-170] should merge to [100-170]
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [100]
    assert df["end"].to_list() == [170]


def test_get_cds_empty_annotation():
    """Test get_cds with annotation containing no CDS features.

    Input: Annotation with only exons
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [150, 250],
            "strand": ["+", "+"],
            "feature": ["exon", "exon"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_cds(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_cds_gbkey_fallback():
    """Test get_cds uses gbkey when feature is not "CDS".

    Input: Annotation with gene name as feature but gbkey="CDS" (C. elegans style)
    Output: GenomicSet containing the CDS region
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 300],
            "end": [200, 400],
            "strand": ["+", "+"],
            "feature": ["Y74C9A.3", "exon"],  # Gene name instead of "CDS"
            "attribute": [
                'transcript_id "trans1"; gbkey "CDS"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_cds(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [100]
    assert df["end"].to_list() == [200]


# read_bed_to_pandas and write_pandas_to_bed tests
def test_read_bed_to_pandas(tmp_path):
    """Test read_bed_to_pandas with basic BED file.

    Input: BED file with 3 intervals
    Output: DataFrame with columns [chrom, start, end]
    """
    bed_file = tmp_path / "test.bed"
    bed_content = """chr1\t100\t200
chr2\t300\t400
chr3\t500\t600
"""
    bed_file.write_text(bed_content)

    result = read_bed_to_pandas(str(bed_file))

    assert len(result) == 3
    assert list(result.columns) == ["chrom", "start", "end"]
    assert result["chrom"].tolist() == ["chr1", "chr2", "chr3"]
    assert result["start"].tolist() == [100, 300, 500]
    assert result["end"].tolist() == [200, 400, 600]


def test_write_pandas_to_bed(tmp_path):
    """Test write_pandas_to_bed writes correct BED format.

    Input: DataFrame with intervals
    Output: BED file with tab-separated values, no header, no index
    """
    bed_file = tmp_path / "output.bed"
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [100, 300],
            "end": [200, 400],
        }
    )

    write_pandas_to_bed(df, str(bed_file))

    # Read back and verify
    content = bed_file.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "chr1\t100\t200"
    assert lines[1] == "chr2\t300\t400"


def test_read_write_bed_roundtrip(tmp_path):
    """Test that reading and writing BED files preserves data.

    Input: DataFrame -> write to BED -> read back
    Output: Original DataFrame matches read DataFrame
    """
    bed_file = tmp_path / "roundtrip.bed"
    original = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [100, 300, 500],
            "end": [200, 400, 600],
        }
    )

    write_pandas_to_bed(original, str(bed_file))
    result = read_bed_to_pandas(str(bed_file))

    pd.testing.assert_frame_equal(result, original)


# load_fasta tests
def test_load_fasta_basic(tmp_path):
    """Test load_fasta with basic FASTA file.

    Input: FASTA file with 2 sequences
    Output: Series with sequence IDs as index
    """
    fasta_file = tmp_path / "test.fasta"
    fasta_content = """>seq1
ATGCATGC
>seq2
GCTAGCTA
"""
    fasta_file.write_text(fasta_content)

    result = load_fasta(str(fasta_file))

    assert len(result) == 2
    assert result.name == "seq"
    assert result.index.tolist() == ["seq1", "seq2"]
    assert result["seq1"] == "ATGCATGC"
    assert result["seq2"] == "GCTAGCTA"


def test_load_fasta_multiline_sequences(tmp_path):
    """Test load_fasta with multi-line sequences.

    Input: FASTA file with sequences split across multiple lines
    Output: Series with concatenated sequences
    """
    fasta_file = tmp_path / "test.fasta"
    fasta_content = """>seq1
ATGC
ATGC
>seq2
GCTA
GCTA
"""
    fasta_file.write_text(fasta_content)

    result = load_fasta(str(fasta_file))

    assert result["seq1"] == "ATGCATGC"
    assert result["seq2"] == "GCTAGCTA"


def test_load_fasta_empty(tmp_path):
    """Test load_fasta with empty FASTA file.

    Input: Empty FASTA file
    Output: Empty Series
    """
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.write_text("")

    result = load_fasta(str(fasta_file))

    assert len(result) == 0
    assert result.name == "seq"


# add_rc tests
def test_add_rc_basic():
    """Test add_rc with basic DNA sequences.

    Input: DataFrame with 1 sequence
    Output: DataFrame with 2 rows (forward and reverse complement)
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGC"],
        }
    )

    result = add_rc(df)

    assert len(result) == 2
    assert result["id"].tolist() == ["seq1_+", "seq1_-"]
    assert result["seq"].tolist() == ["ATGC", "GCAT"]


def test_add_rc_multiple_sequences():
    """Test add_rc with multiple sequences.

    Input: DataFrame with 2 sequences
    Output: DataFrame with 4 rows (2 forward + 2 reverse complement)
    """
    df = pd.DataFrame(
        {
            "id": ["seq1", "seq2"],
            "seq": ["ATGC", "GGCC"],
        }
    )

    result = add_rc(df)

    assert len(result) == 4
    assert result["id"].tolist() == ["seq1_+", "seq2_+", "seq1_-", "seq2_-"]
    assert result["seq"].tolist() == ["ATGC", "GGCC", "GCAT", "GGCC"]


def test_add_rc_preserves_other_columns():
    """Test that add_rc preserves other columns in DataFrame.

    Input: DataFrame with additional columns
    Output: All columns preserved in both forward and RC
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGC"],
            "quality": [30],
        }
    )

    result = add_rc(df)

    assert len(result) == 2
    assert "quality" in result.columns
    assert result["quality"].tolist() == [30, 30]


def test_add_rc_complex_sequence():
    """Test add_rc with a longer, more complex sequence.

    Input: Longer DNA sequence
    Output: Correct reverse complement
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGCTAGCTAGCTA"],
        }
    )

    result = add_rc(df)

    # Expected reverse complement
    expected_rc = str(Seq("ATGCTAGCTAGCTA").reverse_complement())
    assert result[result["id"] == "seq1_-"]["seq"].iloc[0] == expected_rc


def test_add_rc_original_unchanged():
    """Test that add_rc does not modify the original DataFrame.

    Input: Original DataFrame
    Output: Original DataFrame unchanged after add_rc
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGC"],
        }
    )

    original_len = len(df)
    original_id = df["id"].tolist()

    add_rc(df)

    # Original should be unchanged
    assert len(df) == original_len
    assert df["id"].tolist() == original_id


# get_5_prime_utr tests
def test_get_5_prime_utr_positive_strand():
    """Test get_5_prime_utr for transcript on positive strand.

    Input: mRNA transcript with exon [100-300] and CDS [200-250] on '+' strand
    Output: 5' UTR region [100-200]
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [300, 250],
            "strand": ["+", "+"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_5_prime_utr(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [100]
    assert df["end"].to_list() == [200]


def test_get_5_prime_utr_negative_strand():
    """Test get_5_prime_utr for transcript on negative strand.

    Input: mRNA transcript with exon [100-300] and CDS [150-250] on '-' strand
    Output: 5' UTR region [250-300] (genomically after CDS end)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 150],
            "end": [300, 250],
            "strand": ["-", "-"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_5_prime_utr(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    # For '-' strand, 5' UTR is genomically after CDS end
    assert df["start"].to_list() == [250]
    assert df["end"].to_list() == [300]


def test_get_5_prime_utr_multi_exon():
    """Test get_5_prime_utr with multi-exon transcript.

    Input: mRNA with exons [100-200], [300-500] and CDS [350-450]
    Output: 5' UTR in first exon [100-200] and partial second exon [300-350]
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 300, 350],
            "end": [200, 500, 450],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_5_prime_utr(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 2
    df = result.to_pandas()
    assert df["start"].to_list() == [100, 300]
    assert df["end"].to_list() == [200, 350]


def test_get_5_prime_utr_no_utr():
    """Test get_5_prime_utr when CDS starts at exon boundary.

    Input: mRNA with exon and CDS both starting at position 100
    Output: Empty GenomicSet (no 5' UTR)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 100],
            "end": [300, 250],
            "strand": ["+", "+"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_5_prime_utr(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_5_prime_utr_no_cds():
    """Test get_5_prime_utr for non-coding transcript.

    Input: mRNA exon without any CDS
    Output: Empty GenomicSet (no CDS means no UTR by definition)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; transcript_biotype "mRNA"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_5_prime_utr(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


# get_3_prime_utr tests
def test_get_3_prime_utr_positive_strand():
    """Test get_3_prime_utr for transcript on positive strand.

    Input: mRNA transcript with exon [100-300] and CDS [150-200] on '+' strand
    Output: 3' UTR region [200-300]
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 150],
            "end": [300, 200],
            "strand": ["+", "+"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_3_prime_utr(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [200]
    assert df["end"].to_list() == [300]


def test_get_3_prime_utr_negative_strand():
    """Test get_3_prime_utr for transcript on negative strand.

    Input: mRNA transcript with exon [100-300] and CDS [150-250] on '-' strand
    Output: 3' UTR region [100-150] (genomically before CDS start)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 150],
            "end": [300, 250],
            "strand": ["-", "-"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_3_prime_utr(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    # For '-' strand, 3' UTR is genomically before CDS start
    assert df["start"].to_list() == [100]
    assert df["end"].to_list() == [150]


# get_ncrna_exons tests
def test_get_ncrna_exons_includes_lncrna():
    """Test that get_ncrna_exons includes lncRNA exons.

    Input: Annotation with lnc_RNA exon
    Output: GenomicSet containing the lncRNA exon
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; transcript_biotype "lnc_RNA"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1


def test_get_ncrna_exons_includes_mirna():
    """Test that get_ncrna_exons includes miRNA exons.

    Input: Annotation with miRNA exon
    Output: GenomicSet containing the miRNA exon
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; transcript_biotype "miRNA"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1


def test_get_ncrna_exons_excludes_mrna():
    """Test that get_ncrna_exons excludes mRNA exons.

    Input: Annotation with mRNA exon
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; transcript_biotype "mRNA"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_pseudo_true():
    """Test that get_ncrna_exons excludes exons with pseudo="true".

    Input: Annotation with lncRNA exon marked as pseudo
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "lnc_RNA"; pseudo "true"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_pseudogenic_biotypes():
    """Test that get_ncrna_exons excludes pseudogenic biotypes.

    Input: Annotation with pseudogenic_tRNA exon
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "pseudogenic_tRNA"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_transcript_biotype():
    """Test that get_ncrna_exons excludes transcript_biotype="transcript".

    Input: Annotation with transcript biotype "transcript" (used for pseudogenes)
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; transcript_biotype "transcript"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_primary_transcript():
    """Test that get_ncrna_exons excludes primary_transcript (precursors).

    Input: Annotation with primary_transcript biotype
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "primary_transcript"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_gene_biotype_pseudogene():
    """Test that get_ncrna_exons excludes genes with pseudogene biotype.

    Input: Annotation with gene_biotype containing "pseudogene"
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "tRNA"; gene_biotype "tRNA_pseudogene"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_nmd_candidates():
    """Test that get_ncrna_exons excludes NMD candidate transcripts.

    Input: Annotation with "NMD candidate" in description
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "lnc_RNA"; description "NMD candidate alternative"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_partial():
    """Test that get_ncrna_exons excludes partial annotations.

    Input: Annotation with partial="true"
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "lnc_RNA"; partial "true"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_excludes_low_quality():
    """Test that get_ncrna_exons excludes LOW QUALITY transcripts.

    Input: Annotation with "LOW QUALITY" in product
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "lnc_RNA"; product "LOW QUALITY PROTEIN"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


def test_get_ncrna_exons_gbkey_fallback():
    """Test get_ncrna_exons uses gbkey when transcript_biotype is absent.

    Input: Annotation using gbkey instead of transcript_biotype
    Output: GenomicSet containing the exon
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; gbkey "lnc_RNA"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_ncrna_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1


# get_promoters (from annotation) tests
def test_get_promoters_mrna_only_true():
    """Test get_promoters with mRNA_only=True.

    Input: Annotation with mRNA and ncRNA transcripts
    Output: Only mRNA promoters
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [100, 100, 500, 500],
            "end": [200, 150, 600, 550],
            "strand": ["+", "+", "+", "+"],
            "feature": ["exon", "CDS", "exon", "exon"],
            "attribute": [
                'transcript_id "mRNA1"; transcript_biotype "mRNA"',
                'transcript_id "mRNA1"; transcript_biotype "mRNA"',
                'transcript_id "ncRNA1"; transcript_biotype "lnc_RNA"',
                'transcript_id "ncRNA1"; transcript_biotype "lnc_RNA"',
            ],
            "source": ["test"] * 4,
            "score": ["."] * 4,
            "frame": ["."] * 4,
        }
    )

    result = get_promoters(ann, n_upstream=50, n_downstream=25, mRNA_only=True)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    # mRNA promoter at TSS=100: [100-50, 100+25] = [50, 125]
    assert df["chrom"].to_list() == ["chr1"]
    assert df["start"].to_list() == [50]
    assert df["end"].to_list() == [125]


def test_get_promoters_mrna_only_false():
    """Test get_promoters with mRNA_only=False (default).

    Input: Annotation with mRNA and ncRNA transcripts
    Output: Promoters from both mRNA and ncRNA
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [100, 100, 500],
            "end": [200, 150, 600],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "CDS", "exon"],
            "attribute": [
                'transcript_id "mRNA1"; transcript_biotype "mRNA"',
                'transcript_id "mRNA1"; transcript_biotype "mRNA"',
                'transcript_id "ncRNA1"; transcript_biotype "lnc_RNA"',
            ],
            "source": ["test"] * 3,
            "score": ["."] * 3,
            "frame": ["."] * 3,
        }
    )

    result = get_promoters(ann, n_upstream=50, n_downstream=25, mRNA_only=False)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 2


def test_get_promoters_excludes_pseudogene_promoters():
    """Test that get_promoters excludes promoters from pseudogene transcripts.

    Input: Annotation with lncRNA marked as pseudo
    Output: Empty GenomicSet (pseudogene excluded)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "lnc_RNA"; pseudo "true"'
            ],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_promoters(ann, n_upstream=50, n_downstream=25, mRNA_only=False)

    assert isinstance(result, GenomicSet)
    # Pseudogene should be excluded, and there's no mRNA, so empty
    assert result.n_intervals() == 0


# =============================================================================
# Integration Tests with chrY fixture
# =============================================================================
# These tests use the chrY annotation fixture to verify functions work
# correctly with real annotation data.
# Note: chrY_annotation fixture is defined in conftest.py


def test_load_annotation_chrY(chrY_annotation):
    """Integration test: load_annotation with real chrY GTF.

    Verifies that the chrY annotation loads correctly and has expected structure.
    """
    ann = chrY_annotation

    # Should have many features
    assert len(ann) > 10000

    # Check required columns exist
    required_cols = ["chrom", "feature", "start", "end", "strand", "attribute"]
    for col in required_cols:
        assert col in ann.columns

    # All should be on chrY (NC_000024.10)
    assert ann["chrom"].unique().to_list() == ["NC_000024.10"]

    # Should have various feature types
    features = set(ann["feature"].unique().to_list())
    assert "exon" in features
    assert "CDS" in features
    assert "gene" in features
    assert "transcript" in features

    # Start should be less than end (0-based coordinates)
    assert (ann["start"] < ann["end"]).all()


def test_get_mrna_exons_chrY(chrY_annotation):
    """Integration test: get_mrna_exons with real chrY annotation.

    Verifies mRNA exon extraction works with real data.
    """
    result = get_mrna_exons(chrY_annotation)

    # Should have mRNA exons
    assert len(result) > 0

    # Check required columns
    required_cols = ["chrom", "start", "end", "strand", "transcript_id"]
    for col in required_cols:
        assert col in result.columns

    # All on chrY
    assert result["chrom"].unique().to_list() == ["NC_000024.10"]

    # Strands should be valid
    assert set(result["strand"].unique().to_list()).issubset({"+", "-"})

    # All transcript IDs should be non-null
    assert result["transcript_id"].null_count() == 0


def test_get_cds_chrY(chrY_annotation):
    """Integration test: get_cds with real chrY annotation.

    Verifies CDS extraction and merging works with real data.
    """
    result = get_cds(chrY_annotation)

    # Should return a GenomicSet
    assert isinstance(result, GenomicSet)

    # Should have CDS regions
    assert result.n_intervals() > 0

    # Convert to DataFrame for inspection
    df = result.to_polars()
    assert df["chrom"].unique().to_list() == ["NC_000024.10"]


def test_get_ncrna_exons_chrY(chrY_annotation):
    """Integration test: get_ncrna_exons with real chrY annotation.

    Verifies ncRNA exon extraction works with real data.
    """
    result = get_ncrna_exons(chrY_annotation)

    # Should return a GenomicSet
    assert isinstance(result, GenomicSet)

    # chrY should have some ncRNAs (though fewer than autosomes)
    # If none found, that's also valid - just check it doesn't error
    if result.n_intervals() > 0:
        df = result.to_polars()
        assert df["chrom"].unique().to_list() == ["NC_000024.10"]


def test_get_5_prime_utr_chrY(chrY_annotation):
    """Integration test: get_5_prime_utr with real chrY annotation.

    Verifies 5' UTR extraction works with real data.
    """
    result = get_5_prime_utr(chrY_annotation)

    # Should return a GenomicSet
    assert isinstance(result, GenomicSet)

    # Should have 5' UTR regions
    assert result.n_intervals() > 0

    df = result.to_polars()
    assert df["chrom"].unique().to_list() == ["NC_000024.10"]

    # Start should be less than end
    assert (df["start"] < df["end"]).all()


def test_get_3_prime_utr_chrY(chrY_annotation):
    """Integration test: get_3_prime_utr with real chrY annotation.

    Verifies 3' UTR extraction works with real data.
    """
    result = get_3_prime_utr(chrY_annotation)

    # Should return a GenomicSet
    assert isinstance(result, GenomicSet)

    # Should have 3' UTR regions
    assert result.n_intervals() > 0

    df = result.to_polars()
    assert df["chrom"].unique().to_list() == ["NC_000024.10"]

    # Start should be less than end
    assert (df["start"] < df["end"]).all()


def test_get_promoters_chrY(chrY_annotation):
    """Integration test: get_promoters with real chrY annotation.

    Verifies promoter extraction works with real data.
    """
    result = get_promoters(
        chrY_annotation,
        n_upstream=2000,
        n_downstream=500,
        mRNA_only=True,
    )

    # Should return a GenomicSet
    assert isinstance(result, GenomicSet)

    # Should have promoter regions
    assert result.n_intervals() > 0

    df = result.to_polars()
    assert df["chrom"].unique().to_list() == ["NC_000024.10"]


def test_get_promoters_from_exons_chrY(chrY_annotation):
    """Integration test: get_promoters_from_exons with real chrY annotation.

    Verifies promoter extraction from exons works with real data.
    """
    mrna_exons = get_mrna_exons(chrY_annotation)
    result = get_promoters_from_exons(
        mrna_exons,
        n_upstream=2000,
        n_downstream=500,
    )

    # Should return a GenomicSet
    assert isinstance(result, GenomicSet)

    # Should have promoter regions
    assert result.n_intervals() > 0

    df = result.to_polars()
    assert df["chrom"].unique().to_list() == ["NC_000024.10"]


# get_upstream_of_CDS tests
def test_get_upstream_of_CDS_positive_strand():
    """Test get_upstream_of_CDS for transcript on positive strand.

    Input: mRNA transcript with CDS [200-300] on '+' strand
    Output: Upstream region [100-200] (100bp upstream of CDS start)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [400, 300],
            "strand": ["+", "+"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_upstream_of_CDS(ann, dist=100)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [100]
    assert df["end"].to_list() == [200]


def test_get_upstream_of_CDS_negative_strand():
    """Test get_upstream_of_CDS for transcript on negative strand.

    Input: mRNA transcript with CDS [200-300] on '-' strand
    Output: Upstream region [300-400] (100bp upstream of CDS, which is past CDS end)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [400, 300],
            "strand": ["-", "-"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_upstream_of_CDS(ann, dist=100)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [300]
    assert df["end"].to_list() == [400]


def test_get_upstream_of_CDS_with_bounds():
    """Test get_upstream_of_CDS with within_bounds clipping.

    Input: CDS at [200-300] on '+' strand, dist=200, bounds [50-150]
    Output: Should be clipped from [0-200] to [50-150]
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [400, 300],
            "strand": ["+", "+"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    bounds = GenomicSet(pl.DataFrame({"chrom": ["chr1"], "start": [50], "end": [150]}))

    result = get_upstream_of_CDS(ann, dist=200, within_bounds=bounds)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [50]
    assert df["end"].to_list() == [150]


def test_get_upstream_of_CDS_no_cds():
    """Test get_upstream_of_CDS with annotation that has no CDS.

    Input: Annotation with only exons, no CDS
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; transcript_biotype "mRNA"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_upstream_of_CDS(ann, dist=100)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


# get_downstream_of_CDS tests
def test_get_downstream_of_CDS_positive_strand():
    """Test get_downstream_of_CDS for transcript on positive strand.

    Input: mRNA transcript with CDS [200-300] on '+' strand
    Output: Downstream region [300-400] (100bp downstream of CDS end)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [500, 300],
            "strand": ["+", "+"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_downstream_of_CDS(ann, dist=100)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [300]
    assert df["end"].to_list() == [400]


def test_get_downstream_of_CDS_negative_strand():
    """Test get_downstream_of_CDS for transcript on negative strand.

    Input: mRNA transcript with CDS [200-300] on '-' strand
    Output: Downstream region [100-200] (100bp downstream of CDS, before CDS start)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [400, 300],
            "strand": ["-", "-"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_downstream_of_CDS(ann, dist=100)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [100]
    assert df["end"].to_list() == [200]


def test_get_downstream_of_CDS_with_bounds():
    """Test get_downstream_of_CDS with within_bounds clipping.

    Input: CDS at [200-300] on '+' strand, dist=200, bounds [350-450]
    Output: Should be clipped from [300-500] to [350-450]
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [600, 300],
            "strand": ["+", "+"],
            "feature": ["exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans1"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    bounds = GenomicSet(pl.DataFrame({"chrom": ["chr1"], "start": [350], "end": [450]}))

    result = get_downstream_of_CDS(ann, dist=200, within_bounds=bounds)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 1
    df = result.to_pandas()
    assert df["start"].to_list() == [350]
    assert df["end"].to_list() == [450]


def test_get_downstream_of_CDS_no_cds():
    """Test get_downstream_of_CDS with annotation that has no CDS.

    Input: Annotation with only exons, no CDS
    Output: Empty GenomicSet
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [300],
            "strand": ["+"],
            "feature": ["exon"],
            "attribute": ['transcript_id "trans1"; transcript_biotype "mRNA"'],
            "source": ["test"],
            "score": ["."],
            "frame": ["."],
        }
    )

    result = get_downstream_of_CDS(ann, dist=100)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 0


# Integration tests with chrY annotation
def test_get_upstream_of_CDS_chrY(chrY_annotation):
    """Integration test: get_upstream_of_CDS with real chrY annotation.

    Verifies upstream CDS region extraction works with real data.
    """
    result = get_upstream_of_CDS(chrY_annotation, dist=1000)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() > 0

    df = result.to_polars()
    assert df["chrom"].unique().to_list() == ["NC_000024.10"]


def test_get_downstream_of_CDS_chrY(chrY_annotation):
    """Integration test: get_downstream_of_CDS with real chrY annotation.

    Verifies downstream CDS region extraction works with real data.
    """
    result = get_downstream_of_CDS(chrY_annotation, dist=1000)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() > 0

    df = result.to_polars()
    assert df["chrom"].unique().to_list() == ["NC_000024.10"]


def test_get_exons():
    """Test get_exons extracts all exon features regardless of transcript type.

    Input: Annotation with exon, CDS, and gene features
    Output: GenomicSet with only exon features (merged)
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [100, 200, 300, 500],
            "end": [150, 250, 350, 600],
            "strand": ["+", "+", "+", "-"],
            "feature": ["exon", "CDS", "exon", "exon"],
            "attribute": [
                'transcript_id "t1"; transcript_biotype "mRNA"',
                'transcript_id "t1"; transcript_biotype "mRNA"',
                'transcript_id "t2"; transcript_biotype "lnc_RNA"',
                'transcript_id "t3"; transcript_biotype "miRNA"',
            ],
            "source": ["test"] * 4,
            "score": ["."] * 4,
            "frame": ["."] * 4,
        }
    )

    result = get_exons(ann)

    assert isinstance(result, GenomicSet)
    assert result.n_intervals() == 3
    assert result.total_size() == 50 + 50 + 100


def test_get_ensembl_functional_exons_excludes_retained_intron():
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 300, 500],
            "end": [200, 400, 600],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "exon", "exon"],
            "attribute": [
                'transcript_id "t1"; transcript_biotype "protein_coding"',
                'transcript_id "t2"; transcript_biotype "retained_intron"',
                'transcript_id "t3"; transcript_biotype "lncRNA"',
            ],
            "source": ["test"] * 3,
            "score": ["."] * 3,
            "frame": ["."] * 3,
        }
    )

    result = get_ensembl_functional_exons(ann)
    assert result.n_intervals() == 2
    assert result.total_size() == 200


def test_get_ensembl_functional_exons_excludes_all_problematic_biotypes():
    biotypes = [
        "retained_intron",
        "nonsense_mediated_decay",
        "protein_coding_CDS_not_defined",
        "processed_pseudogene",
        "TEC",
        "artifact",
        "non_stop_decay",
    ]
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"] * (len(biotypes) + 1),
            "start": [i * 1000 for i in range(len(biotypes) + 1)],
            "end": [i * 1000 + 100 for i in range(len(biotypes) + 1)],
            "strand": ["+"] * (len(biotypes) + 1),
            "feature": ["exon"] * (len(biotypes) + 1),
            "attribute": [
                f'transcript_id "t{i}"; transcript_biotype "{bt}"'
                for i, bt in enumerate(biotypes)
            ]
            + ['transcript_id "tgood"; transcript_biotype "protein_coding"'],
            "source": ["test"] * (len(biotypes) + 1),
            "score": ["."] * (len(biotypes) + 1),
            "frame": ["."] * (len(biotypes) + 1),
        }
    )

    all_exons = get_exons(ann)
    functional = get_ensembl_functional_exons(ann)

    assert all_exons.n_intervals() == len(biotypes) + 1
    assert functional.n_intervals() == 1


def test_get_exons_for_masking_with_biotype():
    """With transcript_biotype available, behaves like get_ensembl_functional_exons."""
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 300, 500],
            "end": [200, 400, 600],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "exon", "exon"],
            "attribute": [
                'transcript_id "t1"; transcript_biotype "protein_coding"',
                'transcript_id "t2"; transcript_biotype "retained_intron"',
                'transcript_id "t3"; transcript_biotype "lncRNA"',
            ],
            "source": ["test"] * 3,
            "score": ["."] * 3,
            "frame": ["."] * 3,
        }
    )
    result = get_exons_for_masking(ann)
    # retained_intron excluded → 2 exons kept
    assert result.n_intervals() == 2
    assert result.total_size() == 200


def test_get_exons_for_masking_without_biotype():
    """Without transcript_biotype (NCBI GTFs), falls back to all exons."""
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 300, 500],
            "end": [200, 400, 600],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "exon", "exon"],
            "attribute": [
                'gene_id "g1"; gbkey "mRNA"',
                'gene_id "g2"; gbkey "mRNA"',
                'gene_id "g3"; gbkey "ncRNA"',
            ],
            "source": ["test"] * 3,
            "score": ["."] * 3,
            "frame": ["."] * 3,
        }
    )
    result = get_exons_for_masking(ann)
    # No biotype info → all 3 exons returned
    assert result.n_intervals() == 3
    assert result.total_size() == 300


def test_get_exons_for_masking_mixed_biotype():
    """Mixed annotations: exons without biotype must be kept (not silently dropped).

    Most exons have biotype info → Ensembl filtering applies. Exons without
    biotype should still be kept (treated as not in the excluded list), not
    silently dropped due to null propagation through ~biotype.is_in(...).
    """
    # 5 exons: 4 with biotype (1 excluded retained_intron, 3 kept), 1 without
    # null_count=1 < 5//2=2 → Ensembl path triggered
    ann = pl.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "start": [100, 300, 500, 700, 900],
            "end": [200, 400, 600, 800, 1000],
            "strand": ["+"] * 5,
            "feature": ["exon"] * 5,
            "attribute": [
                'transcript_id "t1"; transcript_biotype "protein_coding"',
                'transcript_id "t2"; transcript_biotype "retained_intron"',
                'transcript_id "t3"; transcript_biotype "lncRNA"',
                'transcript_id "t4"; transcript_biotype "miRNA"',
                'transcript_id "t5"; gbkey "ncRNA"',  # no transcript_biotype
            ],
            "source": ["test"] * 5,
            "score": ["."] * 5,
            "frame": ["."] * 5,
        }
    )
    result = get_exons_for_masking(ann)
    # Excluded: retained_intron (1)
    # Kept: protein_coding, lncRNA, miRNA, no-biotype = 4
    # Without the null fix, the no-biotype exon would be dropped, giving 3
    assert result.n_intervals() == 4
    assert result.total_size() == 400


def test_get_exons_for_masking_empty():
    """Empty annotation returns empty GenomicSet."""
    ann = pl.DataFrame(
        {
            "chrom": pl.Series([], dtype=pl.String),
            "start": pl.Series([], dtype=pl.Int64),
            "end": pl.Series([], dtype=pl.Int64),
            "strand": pl.Series([], dtype=pl.String),
            "feature": pl.Series([], dtype=pl.String),
            "attribute": pl.Series([], dtype=pl.String),
            "source": pl.Series([], dtype=pl.String),
            "score": pl.Series([], dtype=pl.String),
            "frame": pl.Series([], dtype=pl.String),
        }
    )
    result = get_exons_for_masking(ann)
    assert result.n_intervals() == 0
