import pandas as pd
import polars as pl
import pytest

from marin_dna.data.intervals import GenomicList, GenomicSet


# GenomicSet tests
def test_genomic_set_initialization_non_overlapping():
    """Test GenomicSet initialization with non-overlapping intervals.

    Input: chr1:0-50, chr1:100-150, chr2:0-200
    Output: chr1:0-50, chr1:100-150, chr2:0-200 (sorted, non-overlapping maintained)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)

    # Should maintain the same intervals since they're non-overlapping, sorted
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 100, 0],
                "end": [50, 150, 200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_merges_overlapping_intervals():
    """Test that GenomicSet merges overlapping intervals on initialization.

    Input: chr1:0-50, chr1:40-60, chr1:90-150
    Output: chr1:0-60, chr1:90-150 (overlapping intervals merged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 40, 90],
            "end": [50, 60, 150],
        }
    )
    gs = GenomicSet(data)

    # Overlapping intervals [0,50] and [40,60] should merge to [0,60]
    # [90,150] should remain separate
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 90],
                "end": [60, 150],
            }
        )
    )
    assert gs == expected


def test_genomic_set_merges_adjacent_intervals():
    """Test that GenomicSet merges adjacent intervals on initialization.

    Input: chr1:0-100, chr1:100-200 (adjacent: end of first equals start of second)
    Output: chr1:0-200 (adjacent intervals merged into single interval)

    Note: Adjacent intervals (where end of one equals start of the next) are merged
    because they form a continuous region.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [100, 200],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_union():
    """Test union operation (|) between two GenomicSets.

    Input set 1: chr1:0-50, chr1:100-150 (two separate intervals with gap 50-100)
    Input set 2: chr1:40-80, chr2:0-200
    Output: chr1:0-80, chr1:100-150, chr2:0-200 (overlaps merged)

    Note: chr1:0-50 and chr1:40-80 merge to chr1:0-80. chr1:100-150 remains separate.
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [40, 0],
            "end": [80, 200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 | gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 100, 0],
                "end": [80, 150, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_intersection():
    """Test intersection operation (&) between two GenomicSets.

    Input set 1: chr1:0-50, chr1:100-150
    Input set 2: chr1:25-75, chr1:120-200
    Output: chr1:25-50, chr1:120-150 (only overlapping regions)

    Note: chr1:0-50 and chr1:100-150 are separate intervals (not merged due to gap).
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [25, 120],
            "end": [75, 200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 & gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [25, 120],
                "end": [50, 150],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtraction():
    """Test subtraction operation (-) between two GenomicSets.

    Input set 1: chr1:0-200 (single merged interval)
    Input set 2: chr1:40-60
    Output: chr1:0-40, chr1:60-200 (subtracted region removed, creating two intervals)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [200],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [40],
            "end": [60],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 60],
                "end": [40, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_union_no_overlap():
    """Test union with completely non-overlapping intervals.

    Input set 1: chr1:0-50
    Input set 2: chr2:100-200
    Output: chr1:0-50, chr2:100-200 (all intervals included, no merging)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr2"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 | gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [0, 100],
                "end": [50, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_intersection_no_overlap():
    """Test intersection with no overlapping intervals.

    Input set 1: chr1:0-50
    Input set 2: chr1:100-200
    Output: (empty set - no overlap)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 & gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtraction_no_overlap():
    """Test subtraction when there's no overlap.

    Input set 1: chr1:0-50
    Input set 2: chr1:100-200
    Output: chr1:0-50 (unchanged, no overlap to subtract)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [50],
            }
        )
    )
    assert result == expected


def test_genomic_set_different_chromosomes():
    """Test operations with intervals on different chromosomes.

    Input set 1: chr1:0-100, chr2:0-100
    Input set 2: chr1:50-150, chr3:50-150

    Union output: chr1:0-150, chr2:0-100, chr3:50-150 (all chromosomes)
    Intersection output: chr1:50-100 (only chr1 overlaps)
    Subtraction output: chr1:0-50, chr2:0-100 (chr1 overlap removed, chr2 unchanged)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 0],
            "end": [100, 100],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr3"],
            "start": [50, 50],
            "end": [150, 150],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)

    union_result = gs1 | gs2
    expected_union = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2", "chr3"],
                "start": [0, 0, 50],
                "end": [150, 100, 150],
            }
        )
    )
    assert union_result == expected_union

    intersection_result = gs1 & gs2
    expected_intersection = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [50],
                "end": [100],
            }
        )
    )
    assert intersection_result == expected_intersection

    subtraction_result = gs1 - gs2
    expected_subtraction = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [0, 0],
                "end": [50, 100],
            }
        )
    )
    assert subtraction_result == expected_subtraction


def test_genomic_set_to_pandas():
    """Test conversion to pandas DataFrame.

    Input: chr1:0-50, chr2:100-200
    Output: Same intervals as pandas DataFrame
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)
    df = gs.to_pandas()

    expected = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    pd.testing.assert_frame_equal(df, expected)
    # Also verify equality works
    assert gs == GenomicSet(expected)


def test_genomic_set_empty():
    """Test GenomicSet with empty DataFrame.

    Input: (empty set)
    Output: (empty set)
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert gs == expected


def test_genomic_set_repr():
    """Test string representation of GenomicSet.

    Input: chr1:0-100
    Output: String representation containing "GenomicSet"
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    repr_str = repr(gs)

    assert "GenomicSet" in repr_str


def test_genomic_set_adjacent_intervals():
    """Test that adjacent intervals (touching but not overlapping) are handled correctly.

    Input: chr1:0-50, chr1:50-100 (adjacent/touching intervals)
    Output: chr1:0-100 (adjacent intervals merged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 50],
            "end": [50, 100],
        }
    )
    gs = GenomicSet(data)

    # Adjacent intervals should be merged by bioframe
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [100],
            }
        )
    )
    assert gs == expected


def test_genomic_set_single_interval():
    """Test GenomicSet with a single interval.

    Input: chr1:100-200
    Output: chr1:100-200 (unchanged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [100],
                "end": [200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_self_union():
    """Test union of a GenomicSet with itself.

    Input: chr1:0-100 | chr1:0-100
    Output: chr1:0-100 (union with itself equals original)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs | gs

    assert result == gs


def test_genomic_set_self_intersection():
    """Test intersection of a GenomicSet with itself.

    Input: chr1:0-100 & chr1:0-100
    Output: chr1:0-100 (intersection with itself equals original)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs & gs

    assert result == gs


def test_genomic_set_self_subtraction():
    """Test subtraction of a GenomicSet from itself.

    Input: chr1:0-100 - chr1:0-100
    Output: (empty set - subtracting itself removes everything)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs - gs

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_equality():
    """Test equality comparison between GenomicSets.

    Input set 1: chr1:0-50, chr2:100-200
    Input set 2: chr1:0-50, chr2:100-200 (same as set 1)
    Input set 3: chr1:0-51, chr2:100-200 (different end)

    Tests:
    - Set 1 == Set 2: True (identical intervals)
    - Set 1 != Set 3: True (different intervals)
    - Set 1 != non-GenomicSet: True (different type)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    data3 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [51, 200],  # Different end value
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    gs3 = GenomicSet(data3)

    assert gs1 == gs2
    assert gs2 == gs1
    assert gs1 != gs3
    assert gs3 != gs1
    assert gs1 != "not a GenomicSet"
    assert gs1 != None  # noqa: E711
    assert gs1 != 123


def test_genomic_set_equality_empty():
    """Test equality with empty GenomicSets.

    Input: (empty set) == (empty set)
    Output: True (empty sets are equal)
    """
    empty1 = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    empty2 = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )

    assert empty1 == empty2
    assert empty2 == empty1


# Invalid input tests
def test_genomic_set_zero_length_interval():
    """Test that GenomicSet accepts zero-length intervals (start == end).

    Input: chr1:50-50 (start equals end, zero-length interval)
    Output: chr1:50-50 (bioframe allows zero-length intervals)

    Note: While conceptually zero-length intervals may seem invalid,
    bioframe's validation only checks that start <= end (not start < end).
    Zero-length intervals are technically valid in bedFrame format.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [50],  # start == end, zero-length interval
        }
    )

    # This should work - bioframe allows start == end
    gs = GenomicSet(data)
    result_df = gs.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    assert result_df.iloc[0]["start"] == 50
    assert result_df.iloc[0]["end"] == 50


def test_genomic_set_invalid_start_greater_than_end():
    """Test that GenomicSet rejects intervals where start > end.

    Input: chr1:100-50 (start > end, invalid)
    Expected: ValueError (invalid bedFrame - starts exceed ends)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [50],  # start > end, invalid
        }
    )

    with pytest.raises(ValueError, match="starts exceed ends"):
        GenomicSet(data)


def test_genomic_set_negative_start():
    """Test that GenomicSet accepts negative start (bioframe allows it).

    Input: chr1:-10-50 (negative start)
    Output: chr1:-10-50 (bioframe doesn't validate non-negative starts)

    Note: While bioframe validates start < end, it doesn't enforce start >= 0.
    Negative starts are technically valid in bioframe's bedFrame format,
    even though the constraint "0 <= start < end" may be conceptually desired.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [-10],  # negative start, but valid for bioframe
            "end": [50],
        }
    )

    # This should work - bioframe doesn't reject negative starts
    gs = GenomicSet(data)
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [-10],
                "end": [50],
            }
        )
    )
    assert gs == expected


def test_genomic_set_invalid_chrom_dtype():
    """Test that GenomicSet rejects invalid chrom dtypes.

    Input: chrom column with int dtype (should be object/string/categorical)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": [1, 2],  # int dtype instead of string
            "start": [0, 100],
            "end": [50, 200],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_start_dtype():
    """Test that GenomicSet rejects invalid start dtype.

    Input: start column with float dtype (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0.5, 100.0],  # float dtype instead of int
            "end": [50, 200],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_end_dtype():
    """Test that GenomicSet rejects invalid end dtype.

    Input: end column with float dtype (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50.5, 200.0],  # float dtype instead of int
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_string_start():
    """Test that GenomicSet rejects string values in start column.

    Input: start column with string values (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": ["0"],  # string instead of int
            "end": [50],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_valid_categorical_chrom():
    """Test that GenomicSet accepts categorical chrom dtype (valid).

    Input: chr1:0-50 with categorical chrom
    Output: chr1:0-50 (categorical is valid dtype for chrom)

    Note: After processing, categorical may be converted to object dtype,
    but the content should be equivalent.
    """
    data = pd.DataFrame(
        {
            "chrom": pd.Categorical(["chr1"]),
            "start": [0],
            "end": [50],
        }
    )

    gs = GenomicSet(data)
    # After merge and processing, check that we have the same intervals
    result_df = gs.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    assert result_df.iloc[0]["start"] == 0
    assert result_df.iloc[0]["end"] == 50


# expand_min_size and add_random_shift tests
def test_genomic_set_expand_min_size_smaller_intervals():
    """Test expand_min_size with intervals smaller than min_size.

    Input: chr1:10-40 (size=30), min_size=50
    Output: chr1:0-50 (expanded equally on both sides to reach min_size=50)

    Note: Interval is expanded equally on both sides to reach min_size.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [10],
            "end": [40],  # size = 30
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=50)

    # Should be expanded: pad = ceil((50-30)/2) = ceil(10) = 10
    # So start = 10-10 = 0, end = 40+10 = 50
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [50],
            }
        )
    )
    assert result == expected


def test_genomic_set_expand_min_size_larger_intervals():
    """Test expand_min_size with intervals already larger than min_size.

    Input: chr1:0-100 (size=100), min_size=50
    Output: chr1:0-100 (unchanged, already larger than min_size)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],  # size = 100, already larger than min_size
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=50)

    # Should be unchanged (pad = max(ceil((50-100)/2), 0) = max(ceil(-25), 0) = 0)
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [100],
            }
        )
    )
    assert result == expected


def test_genomic_set_expand_min_size_causes_overlaps():
    """Test expand_min_size causing overlaps that get merged.

    Input: chr1:10-30, chr1:35-50 (separate, gap 30-35), min_size=30
    Output: chr1:5-58 (expanded intervals overlap and merge)

    Note: After expansion, intervals overlap and should be merged.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [10, 35],
            "end": [30, 50],  # sizes: 20, 15; gap between them
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=30)

    # First: size=20, pad = ceil((30-20)/2) = 5, becomes 5-35
    # Second: size=15, pad = ceil((30-15)/2) = ceil(7.5) = 8, becomes 27-58
    # Now they overlap (35 > 27), so should merge to chr1:5-58
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [5],
                "end": [58],
            }
        )
    )
    assert result == expected


# resize tests


def test_genomic_set_resize_expand_small_interval():
    """Test resize expanding a small interval.

    Input: chr1:100-200 (size=100), target_size=255
    diff=155, left_adj=77, right_adj=78
    Output: chr1:23-278 (size=255)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )
    gs = GenomicSet(data)
    result = gs.resize(target_size=255)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [23],
                "end": [278],
            }
        )
    )
    assert result == expected


def test_genomic_set_resize_shrink_large_interval():
    """Test resize shrinking a large interval.

    Input: chr1:100-450 (size=350), target_size=255
    diff=-95, left_adj=-48, right_adj=-47
    start = 100 - (-48) = 148, end = 450 + (-47) = 403
    Output: chr1:148-403 (size=255)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [450],
        }
    )
    gs = GenomicSet(data)
    result = gs.resize(target_size=255)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [148],
                "end": [403],
            }
        )
    )
    assert result == expected


def test_genomic_set_resize_already_exact_size():
    """Test resize with an interval already at target size.

    Input: chr1:100-355 (size=255), target_size=255
    Output: chr1:100-355 (unchanged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [355],
        }
    )
    gs = GenomicSet(data)
    result = gs.resize(target_size=255)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [100],
                "end": [355],
            }
        )
    )
    assert result == expected


def test_genomic_set_resize_mixed_intervals():
    """Test resize with a mix of small and large intervals.

    Input: chr1:100-200 (size=100), chr2:500-850 (size=350), target_size=255
    chr1: diff=155, left=77, right=78 → 23-278
    chr2: diff=-95, left=-48, right=-47 → 548-803
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [100, 500],
            "end": [200, 850],
        }
    )
    gs = GenomicSet(data)
    result = gs.resize(target_size=255)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [23, 548],
                "end": [278, 803],
            }
        )
    )
    assert result == expected


def test_genomic_set_resize_expansion_causes_overlaps():
    """Test resize causing overlaps that get merged.

    Input: chr1:10-20 (size=10), chr1:25-35 (size=10), target_size=50
    chr1 first: diff=40, left=20, right=20 → -10 to 40
    chr1 second: diff=40, left=20, right=20 → 5 to 55
    Overlap → merged to chr1:-10-55
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [10, 25],
            "end": [20, 35],
        }
    )
    gs = GenomicSet(data)
    result = gs.resize(target_size=50)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [-10],
                "end": [55],
            }
        )
    )
    assert result == expected


def test_genomic_set_resize_empty_set():
    """Test resize on an empty GenomicSet."""
    gs = GenomicSet(pd.DataFrame({"chrom": [], "start": [], "end": []}))
    result = gs.resize(target_size=255)
    expected = GenomicSet(pd.DataFrame({"chrom": [], "start": [], "end": []}))
    assert result == expected


def test_genomic_set_resize_invalid_target_size():
    """Test resize with non-positive target_size raises ValueError."""
    gs = GenomicSet(pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}))
    with pytest.raises(ValueError, match="target_size must be positive"):
        gs.resize(target_size=0)
    with pytest.raises(ValueError, match="target_size must be positive"):
        gs.resize(target_size=-10)


def test_genomic_set_resize_returns_new_instance():
    """Test that resize returns a new GenomicSet instance.

    Input: chr1:10-40, resize(50)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [10],
            "end": [40],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.resize(target_size=50)

    assert gs.to_pandas().equals(original_df)
    assert result != gs
    assert isinstance(result, GenomicSet)


def test_genomic_set_add_flank_basic():
    """Test add_flank with basic interval expansion.

    Input: chr1:50-100 (size=50), flank=10
    Output: chr1:40-110 (expanded by 10 on both sides)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_flank(flank=10)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [40],
                "end": [110],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_flank_multiple_intervals():
    """Test add_flank with multiple non-overlapping intervals.

    Input: chr1:50-100, chr2:200-300, flank=20
    Output: chr1:30-120, chr2:180-320 (both expanded by 20)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [50, 200],
            "end": [100, 300],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_flank(flank=20)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [30, 180],
                "end": [120, 320],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_flank_causes_overlaps():
    """Test add_flank causing overlaps that get merged.

    Input: chr1:50-60, chr1:70-80 (separate, gap 60-70), flank=10
    Output: chr1:40-90 (expanded intervals overlap and merge)

    Note: After expansion, chr1:40-70 and chr1:60-90 overlap, so they merge.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [50, 70],
            "end": [60, 80],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_flank(flank=10)

    # After flank: 40-70 and 60-90, which overlap and merge to 40-90
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [40],
                "end": [90],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_flank_negative_start():
    """Test add_flank with result having negative start values.

    Input: chr1:5-15, flank=10
    Output: chr1:-5-25 (negative start is allowed)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [5],
            "end": [15],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_flank(flank=10)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [-5],
                "end": [25],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_flank_zero():
    """Test add_flank with zero flank (no expansion).

    Input: chr1:50-100, flank=0
    Output: chr1:50-100 (unchanged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_flank(flank=0)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [50],
                "end": [100],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_flank_empty():
    """Test add_flank with empty set.

    Input: (empty set), flank=10
    Output: (empty set)
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_flank(flank=10)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_flank_returns_new_instance():
    """Test that add_flank returns a new GenomicSet instance.

    Input: chr1:50-100, add_flank(10)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.add_flank(flank=10)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different
    assert result != gs
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)


def test_genomic_set_add_random_shift_different_seeds():
    """Test add_random_shift with different seeds produces different results.

    Input: chr1:50-100, seed=42 vs seed=123
    Output: Different shifted positions for each seed
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)

    result1 = gs.add_random_shift(max_shift=10, seed=42)
    result2 = gs.add_random_shift(max_shift=10, seed=123)

    # Results should be different (different random shifts)
    assert result1 != result2

    # Both should have the same interval size (shift preserves size)
    df1 = result1.to_pandas()
    df2 = result2.to_pandas()
    assert len(df1) == 1
    assert len(df2) == 1
    assert (df1.iloc[0]["end"] - df1.iloc[0]["start"]) == (
        df2.iloc[0]["end"] - df2.iloc[0]["start"]
    )
    assert (df1.iloc[0]["end"] - df1.iloc[0]["start"]) == 50  # Original size preserved


def test_genomic_set_add_random_shift_same_seed():
    """Test add_random_shift with same seed produces same results.

    Input: chr1:50-100, seed=42 (twice)
    Output: Same shifted positions for both calls
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)

    result1 = gs.add_random_shift(max_shift=10, seed=42)
    result2 = gs.add_random_shift(max_shift=10, seed=42)

    # Results should be identical (same seed)
    assert result1 == result2


def test_genomic_set_add_random_shift_causes_overlaps():
    """Test add_random_shift causing overlaps that get merged.

    Input: chr1:50-60, chr1:61-70 (adjacent, separate), max_shift=5, seed=42
    Output: Overlapping intervals after shift get merged

    Note: Random shifts may cause intervals to overlap, which should be merged.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [50, 61],
            "end": [60, 70],  # Adjacent intervals
        }
    )
    gs = GenomicSet(data)
    result = gs.add_random_shift(max_shift=5, seed=42)

    # With max_shift=5, intervals could shift and overlap
    # The result should be a valid GenomicSet (merged if overlapping)
    result_df = result.to_pandas()
    assert len(result_df) >= 1  # May be 1 or 2 depending on shifts
    assert all(result_df["chrom"] == "chr1")


def test_genomic_set_add_random_shift_negative_start():
    """Test add_random_shift with negative start values (should be allowed).

    Input: chr1:5-15, max_shift=10, seed that causes negative shift
    Output: chr1 with potentially negative start (allowed)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [5],
            "end": [15],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_random_shift(max_shift=10, seed=999)

    # Should work even if start becomes negative
    result_df = result.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    # Size should be preserved
    assert (result_df.iloc[0]["end"] - result_df.iloc[0]["start"]) == 10


def test_genomic_set_expand_min_size_returns_new_instance():
    """Test that expand_min_size returns a new GenomicSet instance.

    Input: chr1:10-40, expand_min_size(50)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [10],
            "end": [40],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.expand_min_size(min_size=50)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different
    assert result != gs
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)


def test_genomic_set_add_random_shift_returns_new_instance():
    """Test that add_random_shift returns a new GenomicSet instance.

    Input: chr1:50-100, add_random_shift(10, 42)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.add_random_shift(max_shift=10, seed=42)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different (unless shift happened to be 0)
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)


# n_intervals and total_size tests
def test_genomic_set_n_intervals_empty():
    """Test n_intervals with empty set.

    Input: (empty set)
    Output: 0
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 0


def test_genomic_set_n_intervals_single():
    """Test n_intervals with single interval.

    Input: chr1:0-100
    Output: 1
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 1


def test_genomic_set_n_intervals_multiple():
    """Test n_intervals with multiple intervals.

    Input: chr1:0-50, chr1:100-150, chr2:0-200
    Output: 3
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 3


def test_genomic_set_n_intervals_after_merge():
    """Test n_intervals after operations that merge intervals.

    Input: chr1:0-50, chr1:40-60 (overlapping, merge to 1 interval)
    Output: 1 (merged from 2)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    gs = GenomicSet(data)
    # Overlapping intervals should merge to 1
    assert gs.n_intervals() == 1


def test_genomic_set_total_size_empty():
    """Test total_size with empty set.

    Input: (empty set)
    Output: 0
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 0


def test_genomic_set_total_size_single():
    """Test total_size with single interval.

    Input: chr1:0-100 (size=100)
    Output: 100
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 100


def test_genomic_set_total_size_multiple():
    """Test total_size with multiple intervals.

    Input: chr1:0-50 (size=50), chr1:100-150 (size=50), chr2:0-200 (size=200)
    Output: 300 (sum of all sizes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 300


def test_genomic_set_total_size_after_merge():
    """Test total_size after operations that merge intervals.

    Input: chr1:0-50 (size=50), chr1:40-60 (size=20, overlaps with first)
    Output: 60 (merged to chr1:0-60, size=60, less than 50+20=70 due to overlap removal)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    gs = GenomicSet(data)
    # Overlapping intervals merge to chr1:0-60, size=60
    # Original sizes were 50+20=70, but overlap reduces total to 60
    assert gs.total_size() == 60


# Polars support tests
def test_genomic_set_init_from_polars():
    """Test GenomicSet initialization from polars DataFrame.

    Input: polars DataFrame with chr1:0-50, chr1:100-150
    Output: GenomicSet with same intervals
    """
    data = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
        }
    )
    gs = GenomicSet(data)

    assert gs.n_intervals() == 2
    assert gs.total_size() == 100


def test_genomic_set_init_from_polars_merges_overlapping():
    """Test that GenomicSet merges overlapping intervals from polars.

    Input: polars DataFrame with chr1:0-50, chr1:40-60 (overlapping)
    Output: GenomicSet with merged chr1:0-60
    """
    data = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    gs = GenomicSet(data)

    assert gs.n_intervals() == 1
    assert gs.total_size() == 60


def test_genomic_set_init_from_polars_extra_columns():
    """Test GenomicSet from polars DataFrame with extra columns.

    Input: polars DataFrame with chrom, start, end, strand, transcript_id
    Output: GenomicSet using only chrom, start, end columns
    """
    data = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
            "strand": ["+", "-"],
            "transcript_id": ["trans1", "trans2"],
        }
    )
    gs = GenomicSet(data)

    assert gs.n_intervals() == 2
    # Extra columns should be ignored
    result_df = gs.to_pandas()
    assert list(result_df.columns) == ["chrom", "start", "end"]


def test_genomic_set_to_polars():
    """Test conversion to polars DataFrame.

    Input: GenomicSet with chr1:0-50, chr2:100-200
    Output: polars DataFrame with same intervals
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)
    result = gs.to_polars()

    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 3)
    assert result["chrom"].to_list() == ["chr1", "chr2"]
    assert result["start"].to_list() == [0, 100]
    assert result["end"].to_list() == [50, 200]


def test_genomic_set_to_polars_empty():
    """Test to_polars with empty GenomicSet.

    Input: Empty GenomicSet
    Output: Empty polars DataFrame with correct columns
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    result = gs.to_polars()

    assert isinstance(result, pl.DataFrame)
    assert result.shape == (0, 3)
    assert result.columns == ["chrom", "start", "end"]


def test_genomic_set_polars_roundtrip():
    """Test polars -> GenomicSet -> polars roundtrip.

    Input: polars DataFrame
    Output: Same data after roundtrip conversion
    """
    original = pl.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(original)
    result = gs.to_polars()

    assert result["chrom"].to_list() == original["chrom"].to_list()
    assert result["start"].to_list() == original["start"].to_list()
    assert result["end"].to_list() == original["end"].to_list()


# read_bed and write_bed tests
def test_genomic_set_write_bed_read_bed_roundtrip(tmp_path):
    """Test write_bed and read_bed roundtrip.

    Input: GenomicSet with chr1:0-50, chr2:100-200
    Output: Same intervals after write and read
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)

    bed_path = tmp_path / "test.bed"
    gs.write_bed(str(bed_path))

    result = GenomicSet.read_bed(str(bed_path))
    assert result == gs


def test_genomic_set_read_bed_basic(tmp_path):
    """Test read_bed with basic BED file.

    Input: BED file with chr1:10-100, chr2:200-500
    Output: GenomicSet with same intervals
    """
    bed_path = tmp_path / "test.bed"
    bed_path.write_text("chr1\t10\t100\nchr2\t200\t500\n")

    gs = GenomicSet.read_bed(str(bed_path))

    assert gs.n_intervals() == 2
    df = gs.to_pandas()
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["start"] == 10
    assert df.iloc[0]["end"] == 100
    assert df.iloc[1]["chrom"] == "chr2"
    assert df.iloc[1]["start"] == 200
    assert df.iloc[1]["end"] == 500


def test_genomic_set_read_bed_merges_overlapping(tmp_path):
    """Test read_bed merges overlapping intervals.

    Input: BED file with chr1:0-50, chr1:40-60 (overlapping)
    Output: GenomicSet with merged chr1:0-60
    """
    bed_path = tmp_path / "test.bed"
    bed_path.write_text("chr1\t0\t50\nchr1\t40\t60\n")

    gs = GenomicSet.read_bed(str(bed_path))

    assert gs.n_intervals() == 1
    assert gs.total_size() == 60


def test_genomic_set_write_bed_format(tmp_path):
    """Test write_bed produces correct BED format.

    Input: GenomicSet with chr1:0-50, chr2:100-200
    Output: Tab-separated BED file with no header
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)

    bed_path = tmp_path / "test.bed"
    gs.write_bed(str(bed_path))

    content = bed_path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "chr1\t0\t50"
    assert lines[1] == "chr2\t100\t200"


def test_genomic_set_write_bed_empty(tmp_path):
    """Test write_bed with empty GenomicSet.

    Input: Empty GenomicSet
    Output: Empty BED file
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)

    bed_path = tmp_path / "test.bed"
    gs.write_bed(str(bed_path))

    content = bed_path.read_text()
    assert content == ""


# read_parquet and write_parquet tests
def test_genomic_set_write_parquet_read_parquet_roundtrip(tmp_path):
    """Test write_parquet and read_parquet roundtrip.

    Input: GenomicSet with chr1:0-50, chr2:100-200
    Output: Same intervals after write and read
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)

    parquet_path = tmp_path / "test.parquet"
    gs.write_parquet(str(parquet_path))

    result = GenomicSet.read_parquet(str(parquet_path))
    assert result == gs


def test_genomic_set_read_parquet_basic(tmp_path):
    """Test read_parquet with basic parquet file.

    Input: Parquet file with chr1:10-100, chr2:200-500
    Output: GenomicSet with same intervals
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [10, 200],
            "end": [100, 500],
        }
    )
    parquet_path = tmp_path / "test.parquet"
    data.to_parquet(str(parquet_path), index=False)

    gs = GenomicSet.read_parquet(str(parquet_path))

    assert gs.n_intervals() == 2
    df = gs.to_pandas()
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["start"] == 10
    assert df.iloc[0]["end"] == 100


def test_genomic_set_read_parquet_merges_overlapping(tmp_path):
    """Test read_parquet merges overlapping intervals.

    Input: Parquet file with chr1:0-50, chr1:40-60 (overlapping)
    Output: GenomicSet with merged chr1:0-60
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    parquet_path = tmp_path / "test.parquet"
    data.to_parquet(str(parquet_path), index=False)

    gs = GenomicSet.read_parquet(str(parquet_path))

    assert gs.n_intervals() == 1
    assert gs.total_size() == 60


def test_genomic_set_write_parquet_empty(tmp_path):
    """Test write_parquet with empty GenomicSet.

    Input: Empty GenomicSet
    Output: Parquet file with no rows
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)

    parquet_path = tmp_path / "test.parquet"
    gs.write_parquet(str(parquet_path))

    result = GenomicSet.read_parquet(str(parquet_path))
    assert result.n_intervals() == 0


def test_genomic_set_parquet_preserves_dtypes(tmp_path):
    """Test parquet roundtrip preserves column dtypes.

    Input: GenomicSet with string chrom and int start/end
    Output: Same dtypes after roundtrip
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)

    parquet_path = tmp_path / "test.parquet"
    gs.write_parquet(str(parquet_path))
    result = GenomicSet.read_parquet(str(parquet_path))

    result_df = result.to_pandas()
    assert result_df["chrom"].dtype == object
    assert result_df["start"].dtype in [int, "int64"]
    assert result_df["end"].dtype in [int, "int64"]


# filter_size tests
def test_genomic_set_filter_size_min_only():
    """Test filter_size with only min_size specified.

    Input: chr1:0-20 (size=20), chr1:100-150 (size=50), chr2:0-200 (size=200), min_size=30
    Output: chr1:100-150, chr2:0-200 (only intervals >= 30bp)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [20, 150, 200],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=30)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [100, 0],
                "end": [150, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_max_only():
    """Test filter_size with only max_size specified.

    Input: chr1:0-20 (size=20), chr1:100-150 (size=50), chr2:0-300 (size=300), max_size=100
    Output: chr1:0-20, chr1:100-150 (only intervals <= 100bp)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [20, 150, 300],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(max_size=100)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 100],
                "end": [20, 150],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_both_min_and_max():
    """Test filter_size with both min_size and max_size.

    Input: chr1:0-15 (size=15), chr1:50-100 (size=50), chr1:200-350 (size=150), min_size=20, max_size=100
    Output: chr1:50-100 (only interval with 20 <= size <= 100)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 50, 200],
            "end": [15, 100, 350],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=20, max_size=100)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [50],
                "end": [100],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_removes_all():
    """Test filter_size that removes all intervals.

    Input: chr1:0-20 (size=20), chr1:100-150 (size=50), min_size=100
    Output: (empty set - no intervals >= 100bp)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [20, 150],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=100)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_keeps_all():
    """Test filter_size that keeps all intervals.

    Input: chr1:0-50 (size=50), chr1:100-200 (size=100), min_size=10, max_size=200
    Output: chr1:0-50, chr1:100-200 (all intervals within range)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=10, max_size=200)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 100],
                "end": [50, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_empty_set():
    """Test filter_size with empty set.

    Input: (empty set), min_size=20
    Output: (empty set)
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=20)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_exact_boundaries():
    """Test filter_size with intervals exactly matching boundaries.

    Input: chr1:0-20 (size=20), chr1:50-100 (size=50), chr1:200-300 (size=100), min_size=20, max_size=100
    Output: chr1:0-20, chr1:50-100, chr1:200-300 (all intervals, boundaries are inclusive)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 50, 200],
            "end": [20, 100, 300],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=20, max_size=100)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1"],
                "start": [0, 50, 200],
                "end": [20, 100, 300],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_no_parameters():
    """Test filter_size with no parameters (keeps all).

    Input: chr1:0-50, chr1:100-200, min_size=None, max_size=None
    Output: chr1:0-50, chr1:100-200 (all intervals kept)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size()

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 100],
                "end": [50, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_returns_new_instance():
    """Test that filter_size returns a new GenomicSet instance.

    Input: chr1:0-30, chr1:100-200, filter_size(min_size=40)
    Output: New GenomicSet, original unchanged, chr1:0-30 filtered out
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [30, 200],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.filter_size(min_size=40)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different (one interval filtered out)
    assert result != gs
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)
    # Result should have only the large interval
    assert result.n_intervals() == 1


def test_genomic_set_filter_size_multiple_chromosomes():
    """Test filter_size with intervals across multiple chromosomes.

    Input: chr1:0-30 (size=30), chr1:100-200 (size=100), chr2:0-40 (size=40), chr2:100-250 (size=150), min_size=50
    Output: chr1:100-200, chr2:100-250 (only intervals >= 50bp across all chromosomes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [0, 100, 0, 100],
            "end": [30, 200, 40, 250],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=50)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [100, 100],
                "end": [200, 250],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_single_interval():
    """Test filter_size with single interval.

    Input: chr1:0-100 (size=100), min_size=50, max_size=150
    Output: chr1:0-100 (interval within range)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=50, max_size=150)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [100],
            }
        )
    )
    assert result == expected


def test_genomic_set_filter_size_zero_length_interval():
    """Test filter_size with zero-length intervals.

    Input: chr1:50-50 (size=0), chr1:100-150 (size=50), min_size=1
    Output: chr1:100-150 (zero-length interval filtered out)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [50, 100],
            "end": [50, 150],
        }
    )
    gs = GenomicSet(data)
    result = gs.filter_size(min_size=1)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [100],
                "end": [150],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtract_empty_from_nonempty():
    """Test subtracting empty GenomicSet from non-empty GenomicSet.

    Input set 1: chr1:0-100, chr2:0-50
    Input set 2: (empty set)
    Output: chr1:0-100, chr2:0-50 (unchanged when subtracting empty)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 0],
            "end": [100, 50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [0, 0],
                "end": [100, 50],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtract_nonempty_from_empty():
    """Test subtracting non-empty GenomicSet from empty GenomicSet.

    Input set 1: (empty set)
    Input set 2: chr1:0-100
    Output: (empty set - nothing to subtract from)
    """
    data1 = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtract_empty_from_empty():
    """Test subtracting empty GenomicSet from empty GenomicSet.

    Input set 1: (empty set)
    Input set 2: (empty set)
    Output: (empty set)
    """
    data1 = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_filter_not_overlapping_removes_overlapping():
    """Test that filter_not_overlapping removes intervals with any overlap.

    Input set 1: chr1:0-100, chr1:200-300, chr1:400-500
    Input set 2: chr1:50-60, chr1:450-600
    Output: chr1:200-300 (only interval with no overlap kept)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 200, 400],
            "end": [100, 300, 500],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [50, 450],
            "end": [60, 600],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1.filter_not_overlapping(gs2)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [200],
                "end": [300],
            }
        )
    )
    assert result == expected


def test_filter_not_overlapping_no_overlap():
    """Test filter_not_overlapping when no intervals overlap.

    Input set 1: chr1:0-100, chr1:200-300
    Input set 2: chr2:0-100
    Output: chr1:0-100, chr1:200-300 (all kept, different chroms)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 200],
            "end": [100, 300],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr2"],
            "start": [0],
            "end": [100],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1.filter_not_overlapping(gs2)

    assert result == gs1


def test_filter_not_overlapping_empty_other():
    """Test filter_not_overlapping with empty other set returns self unchanged."""
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    empty = pd.DataFrame({"chrom": [], "start": [], "end": []})

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(empty)
    result = gs1.filter_not_overlapping(gs2)

    assert result == gs1


def test_filter_not_overlapping_empty_self():
    """Test filter_not_overlapping with empty self returns empty."""
    empty = pd.DataFrame({"chrom": [], "start": [], "end": []})
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )

    gs1 = GenomicSet(empty)
    gs2 = GenomicSet(data2)
    result = gs1.filter_not_overlapping(gs2)

    assert result == gs1


# GenomicList tests


def test_genomic_list_preserves_overlapping_intervals():
    """Overlapping intervals are NOT merged."""
    data = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 40], "end": [80, 120]})
    gl = GenomicList(data)
    assert len(gl) == 2
    assert gl._data["start"].tolist() == [0, 40]


def test_genomic_list_init_from_polars():
    data = pl.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 40], "end": [80, 120]})
    gl = GenomicList(data)
    assert len(gl) == 2


def test_genomic_list_resize_independent():
    """Two adjacent elements resize independently without merging."""
    data = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [100, 110], "end": [120, 130]}
    )
    gl = GenomicList(data).resize(50)
    assert len(gl) == 2
    assert gl._data["start"].tolist() == [85, 95]
    assert gl._data["end"].tolist() == [135, 145]


def test_genomic_list_drops_extra_columns():
    data = pd.DataFrame(
        {"chrom": ["chr1"], "start": [100], "end": [200], "name": ["enh1"]}
    )
    gl = GenomicList(data)
    assert list(gl._data.columns) == ["chrom", "start", "end"]


def test_genomic_list_resize_invalid():
    gl = GenomicList(pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]}))
    with pytest.raises(ValueError):
        gl.resize(0)


def test_genomic_list_filter_size():
    data = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr1"], "start": [0, 0, 0], "end": [50, 100, 255]}
    )
    gl = GenomicList(data)
    result = gl.filter_size(min_size=100, max_size=255)
    assert len(result) == 2
    assert result._data["end"].tolist() == [100, 255]


def test_genomic_list_filter_not_overlapping():
    """Intervals overlapping an exon set are dropped; others kept."""
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 100, 300],
            "end": [50, 200, 400],
        }
    )
    exons = GenomicSet(pd.DataFrame({"chrom": ["chr1"], "start": [120], "end": [150]}))
    result = GenomicList(intervals).filter_not_overlapping(exons)
    assert len(result) == 2
    assert result._data["start"].tolist() == [0, 300]


def test_genomic_list_filter_within():
    """Only intervals fully inside defined regions are kept."""
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [10, 50, 200],
            "end": [30, 120, 250],
        }
    )
    defined = GenomicSet(
        pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 200], "end": [100, 300]})
    )
    result = GenomicList(intervals).filter_within(defined)
    # 10-30 is within 0-100  ✓
    # 50-120 extends past 100 ✗
    # 200-250 is within 200-300 ✓
    assert len(result) == 2
    assert result._data["start"].tolist() == [10, 200]


def test_genomic_list_filter_within_partial_coverage():
    """Interval spanning a gap in defined regions is dropped."""
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [50], "end": [250]})
    defined = GenomicSet(
        pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 200], "end": [100, 300]})
    )
    result = GenomicList(intervals).filter_within(defined)
    # 50-250 spans the gap 100-200, coverage = 50 + 50 = 100 < 200
    assert len(result) == 0


def test_genomic_list_filter_within_empty():
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [10], "end": [30]})
    empty_regions = GenomicSet(pd.DataFrame({"chrom": [], "start": [], "end": []}))
    result = GenomicList(intervals).filter_within(empty_regions)
    assert len(result) == 0


def test_genomic_list_write_bed(tmp_path):
    data = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [10, 20], "end": [50, 80]})
    path = str(tmp_path / "out.bed")
    GenomicList(data).write_bed(path)
    written = pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"])
    assert written["chrom"].tolist() == ["chr1", "chr2"]
    assert len(written.columns) == 3


def test_genomic_list_equality():
    data = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    assert GenomicList(data) == GenomicList(data.copy())
    assert GenomicList(data) != "not a GenomicList"


def test_genomic_list_adjacent_enhancers_survive_resize():
    """Reproduces the bug: two enhancers 6bp apart should both survive
    resize to 255bp as independent elements."""
    data = pd.DataFrame(
        {
            "chrom": ["chr19", "chr19"],
            "start": [30360805, 30361145],
            "end": [30361139, 30361298],
        }
    )
    gl = GenomicList(data).resize(255)
    result = gl.filter_size(min_size=255, max_size=255)
    assert len(result) == 2
