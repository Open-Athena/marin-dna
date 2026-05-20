"""Tests for feature-based sample matching utilities.

Ported from TraitGym (commit e59d612e9; tests/test_matching.py).
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from marin_dna.pipelines.evals.matching import (
    BIN_NA,
    BIN_OOR,
    EXON_DIST_BIN_EDGES,
    MAF_BIN_EDGES,
    MAF_BIN_EDGES_5,
    MAF_BIN_EDGES_20,
    MAF_TIERED_V1,
    MATCH_GROUP_COL,
    TSS_DIST_BIN_EDGES,
    _combine_results,
    _find_closest,
    _match_single_group,
    _scale_features,
    _sort_by_coordinates,
    _validate_columns,
    add_subset_distance_bins,
    add_subset_distance_bins_v2,
    add_tiered_maf_bin,
    bin_feature,
    match_features,
)


def make_variants(
    n: int,
    chrom: str = "1",
    start_pos: int = 100,
    seed: int = 42,
    **extra_cols: list,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    nucs = ["A", "C", "G", "T"]
    data = {
        "chrom": [chrom] * n,
        "pos": list(range(start_pos, start_pos + n)),
        "ref": rng.choice(nucs, n).tolist(),
        "alt": rng.choice(nucs, n).tolist(),
    }
    data.update(extra_cols)
    return pl.DataFrame(data)


class TestValidateColumns:
    def test_passes_when_all_columns_present(self) -> None:
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        _validate_columns(df, ["a", "b"], "test")

    def test_raises_on_missing_columns(self) -> None:
        df = pl.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing columns"):
            _validate_columns(df, ["a", "c", "d"], "test")

    def test_error_message_includes_dataframe_name(self) -> None:
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="my_df is missing"):
            _validate_columns(df, ["a", "b"], "my_df")


class TestScaleFeatures:
    def test_returns_scaled_columns(self) -> None:
        pos = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [10, 20, 30]})
        neg = pd.DataFrame({"feat1": [4, 5, 6], "feat2": [40, 50, 60]})
        pos_scaled, neg_scaled, scaled_cols = _scale_features(
            pos, neg, ["feat1", "feat2"]
        )

        assert scaled_cols == ["feat1_scaled", "feat2_scaled"]
        assert "feat1_scaled" in pos_scaled.columns
        assert "feat2_scaled" in neg_scaled.columns

    def test_preserves_original_columns(self) -> None:
        pos = pd.DataFrame({"feat1": [1, 2, 3], "other": ["a", "b", "c"]})
        neg = pd.DataFrame({"feat1": [4, 5, 6], "other": ["d", "e", "f"]})
        pos_scaled, _, _ = _scale_features(pos, neg, ["feat1"])

        assert "feat1" in pos_scaled.columns
        assert "other" in pos_scaled.columns
        pd.testing.assert_series_equal(pos_scaled["feat1"], pos["feat1"])

    def test_robust_scaling_uses_median_and_iqr(self) -> None:
        pos = pd.DataFrame({"feat1": [0.0, 1.0, 2.0]})
        neg = pd.DataFrame({"feat1": [3.0, 4.0, 5.0]})
        pos_scaled, _, _ = _scale_features(pos, neg, ["feat1"])

        all_original = np.array([0, 1, 2, 3, 4, 5])
        median = np.median(all_original)
        q1, q3 = np.percentile(all_original, [25, 75])
        iqr = q3 - q1
        expected_pos = (np.array([0, 1, 2]) - median) / iqr
        np.testing.assert_array_almost_equal(
            pos_scaled["feat1_scaled"].values, expected_pos
        )


class TestFindClosest:
    def test_finds_k_closest(self) -> None:
        pos = pd.DataFrame({"x": [0.0], "y": [0.0]})
        neg = pd.DataFrame(
            {
                "x": [1.0, 2.0, 10.0, 3.0],
                "y": [1.0, 2.0, 10.0, 3.0],
                "id": ["a", "b", "c", "d"],
            }
        )
        result = _find_closest(pos, neg, ["x", "y"], k=2)

        assert len(result) == 2
        assert set(result["id"].tolist()) == {"a", "b"}

    def test_no_replacement_across_positives(self) -> None:
        pos = pd.DataFrame({"x": [0.0, 0.1]})
        neg = pd.DataFrame({"x": [0.0, 0.2, 0.3, 0.4], "id": [0, 1, 2, 3]})
        result = _find_closest(pos, neg, ["x"], k=2)

        assert len(result) == 4
        assert len(set(result["id"].tolist())) == 4

    def test_euclidean_distance_used(self) -> None:
        pos = pd.DataFrame({"x": [0.0], "y": [0.0]})
        neg = pd.DataFrame(
            {
                "x": [3.0, 0.0],
                "y": [0.0, 4.0],
                "id": ["three", "four"],
            }
        )
        result = _find_closest(pos, neg, ["x", "y"], k=1)

        assert result["id"].iloc[0] == "three"


class TestMatchSingleGroup:
    def test_matches_within_group(self) -> None:
        pos = pd.DataFrame({"cat": ["A"], "feat": [1.0]}).set_index("cat")
        neg = pd.DataFrame({"cat": ["A", "A"], "feat": [1.1, 2.0]}).set_index("cat")
        result = _match_single_group(pos, neg, "A", ["feat"], k=1, seed=42)

        assert result is not None
        pos_out, neg_out = result
        assert len(pos_out) == 1
        assert len(neg_out) == 1

    def test_warns_and_subsamples_on_insufficient_negatives(self) -> None:
        pos = pd.DataFrame({"cat": ["A"] * 5, "feat": [1.0] * 5}).set_index("cat")
        neg = pd.DataFrame({"cat": ["A"] * 3, "feat": [1.0, 2.0, 3.0]}).set_index("cat")

        with pytest.warns(UserWarning, match="Insufficient negatives"):
            result = _match_single_group(pos, neg, "A", ["feat"], k=2, seed=42)

        assert result is not None
        pos_out, neg_out = result
        assert len(pos_out) == 1
        assert len(neg_out) == 2

    def test_random_sampling_when_no_continuous_features(self) -> None:
        pos = pd.DataFrame({"cat": ["A"] * 2}).set_index("cat")
        neg = pd.DataFrame({"cat": ["A"] * 10, "id": list(range(10))}).set_index("cat")
        result = _match_single_group(pos, neg, "A", [], k=3, seed=42)

        assert result is not None
        pos_out, neg_out = result
        assert len(neg_out) == 6

    def test_drops_all_positives_when_n_neg_below_k(self) -> None:
        """At k=9 a stratum with < 9 negatives subsamples positives to
        floor(n_neg / k) = 0. `_find_closest` must handle the empty-pos
        case rather than tripping the n_neg < k assertion."""
        pos = pd.DataFrame({"cat": ["A"] * 3, "feat": [1.0, 2.0, 3.0]}).set_index("cat")
        neg = pd.DataFrame(
            {"cat": ["A"] * 5, "feat": [10.0, 11.0, 12.0, 13.0, 14.0]}
        ).set_index("cat")
        with pytest.warns(UserWarning, match="Insufficient negatives"):
            result = _match_single_group(pos, neg, "A", ["feat"], k=9, seed=42)
        assert result is not None
        pos_out, neg_out = result
        assert len(pos_out) == 0
        assert len(neg_out) == 0


class TestCombineResults:
    def test_assigns_match_group(self) -> None:
        pos1 = pd.DataFrame({"x": [1, 2]})
        neg1 = pd.DataFrame({"x": [10, 11, 12, 13]})
        result = _combine_results([pos1], [neg1], k=2)

        assert MATCH_GROUP_COL in result.columns
        assert set(result[MATCH_GROUP_COL].unique()) == {0, 1}
        assert (result[MATCH_GROUP_COL] == 0).sum() == 3
        assert (result[MATCH_GROUP_COL] == 1).sum() == 3

    def test_empty_lists_return_empty_dataframe(self) -> None:
        result = _combine_results([], [], k=2)
        assert len(result) == 0

    def test_multiple_groups_combined(self) -> None:
        pos1 = pd.DataFrame({"x": [1]})
        pos2 = pd.DataFrame({"x": [2]})
        neg1 = pd.DataFrame({"x": [10, 11]})
        neg2 = pd.DataFrame({"x": [20, 21]})
        result = _combine_results([pos1, pos2], [neg1, neg2], k=2)

        assert len(result) == 6


class TestSortByCoordinates:
    def test_sorts_by_chrom_pos_ref_alt(self) -> None:
        df = pl.DataFrame(
            {
                "chrom": ["2", "1", "1"],
                "pos": [100, 200, 100],
                "ref": ["A", "A", "A"],
                "alt": ["T", "T", "G"],
            }
        )
        result = _sort_by_coordinates(df)

        assert result["chrom"].to_list() == ["1", "1", "2"]
        assert result["pos"].to_list() == [100, 200, 100]
        assert result["alt"].to_list() == ["G", "T", "T"]


class TestMatchFeatures:
    def test_output_has_match_group_column(self) -> None:
        pos = make_variants(3, category=["A", "A", "A"], feat=[1.0, 2.0, 3.0])
        neg = make_variants(
            9, chrom="1", start_pos=200, category=["A"] * 9, feat=list(range(9))
        )
        result = match_features(pos, neg, ["feat"], ["category"], k=2)

        assert MATCH_GROUP_COL in result.columns

    def test_correct_row_count(self) -> None:
        n_pos = 4
        k = 3
        pos = make_variants(
            n_pos, category=["A"] * n_pos, feat=[float(i) for i in range(n_pos)]
        )
        neg = make_variants(
            n_pos * k,
            chrom="1",
            start_pos=200,
            category=["A"] * (n_pos * k),
            feat=list(range(n_pos * k)),
        )
        result = match_features(pos, neg, ["feat"], ["category"], k=k)

        assert len(result) == n_pos + n_pos * k

    def test_sorted_by_coordinates(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["2", "1"],
                "pos": [100, 50],
                "ref": ["A", "C"],
                "alt": ["T", "G"],
                "cat": ["A", "A"],
                "feat": [1.0, 2.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1", "2", "2"],
                "pos": [60, 70, 110, 120],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                "cat": ["A", "A", "A", "A"],
                "feat": [2.1, 2.2, 1.1, 1.2],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat"], k=2)

        chroms = result["chrom"].to_list()
        assert chroms[0] == "1"

    def test_reproducibility_with_seed(self) -> None:
        pos = make_variants(2, category=["A", "A"], feat=[1.0, 2.0])
        neg = make_variants(
            10, chrom="1", start_pos=200, category=["A"] * 10, feat=list(range(10))
        )

        result1 = match_features(pos, neg, ["feat"], ["category"], k=2, seed=123)
        result2 = match_features(pos, neg, ["feat"], ["category"], k=2, seed=123)

        assert result1.equals(result2)

    def test_different_seeds_give_different_results(self) -> None:
        pos = make_variants(2, category=["A", "A"])
        neg = make_variants(20, chrom="1", start_pos=200, category=["A"] * 20)

        result1 = match_features(pos, neg, [], ["category"], k=3, seed=1)
        result2 = match_features(pos, neg, [], ["category"], k=3, seed=2)

        assert not result1.equals(result2)

    def test_closest_matching_behavior(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1"],
                "pos": [100],
                "ref": ["A"],
                "alt": ["T"],
                "cat": ["A"],
                "feat": [5.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1", "1"],
                "pos": [200, 300, 400],
                "ref": ["A", "A", "A"],
                "alt": ["T", "T", "T"],
                "cat": ["A", "A", "A"],
                "feat": [5.1, 100.0, 5.2],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat"], k=2)

        matched_positions = result["pos"].to_list()
        assert 100 in matched_positions
        assert 200 in matched_positions
        assert 400 in matched_positions
        assert 300 not in matched_positions


class TestMatchFeaturesEdgeCases:
    def test_empty_continuous_features_does_random_sampling(self) -> None:
        pos = make_variants(2, category=["A", "A"])
        neg = make_variants(10, chrom="1", start_pos=200, category=["A"] * 10)
        result = match_features(pos, neg, [], ["category"], k=2)

        assert len(result) == 2 + 2 * 2

    def test_empty_categorical_features_raises(self) -> None:
        # match_features needs at least one categorical key (partition_by and
        # the iter-33 semi-join both require it). Make sure the failure mode
        # is a clear AssertionError, not a buried polars-internal exception.
        pos = make_variants(2, feat=[1.0, 5.0])
        neg = make_variants(
            5, chrom="1", start_pos=200, feat=[1.1, 1.2, 5.1, 5.2, 99.0]
        )
        with pytest.raises(AssertionError, match="categorical_features"):
            match_features(pos, neg, ["feat"], [], k=1)

    def test_multiple_categorical_features(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [100, 200],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat1": ["A", "B"],
                "cat2": ["X", "Y"],
                "feat": [1.0, 2.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1", "1", "1"],
                "pos": [300, 400, 500, 600],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                "cat1": ["A", "A", "B", "B"],
                "cat2": ["X", "X", "Y", "Y"],
                "feat": [1.1, 1.2, 2.1, 2.2],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat1", "cat2"], k=2)

        assert len(result) == 2 + 2 * 2

    def test_warns_on_insufficient_negatives(self) -> None:
        pos = make_variants(5, category=["A"] * 5, feat=[float(i) for i in range(5)])
        neg = make_variants(
            3, chrom="1", start_pos=200, category=["A"] * 3, feat=[0.0, 1.0, 2.0]
        )

        with pytest.warns(UserWarning, match="Insufficient negatives"):
            result = match_features(pos, neg, ["feat"], ["category"], k=2)

        assert len(result) == 1 + 1 * 2

    def test_warns_on_missing_category(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [100, 200],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat": ["A", "B"],
                "feat": [1.0, 2.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [300, 400],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat": ["A", "A"],
                "feat": [1.1, 1.2],
            }
        )

        with pytest.warns(UserWarning, match="No negatives found"):
            result = match_features(pos, neg, ["feat"], ["cat"], k=2)

        assert len(result) == 1 + 1 * 2

    def test_missing_columns_raises_value_error(self) -> None:
        pos = pl.DataFrame({"chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["T"]})
        neg = pos.clone()

        with pytest.raises(ValueError, match="pos is missing columns"):
            match_features(pos, neg, ["missing_feat"], [], k=1)

    def test_missing_coordinates_raises_value_error(self) -> None:
        pos = pl.DataFrame({"chrom": ["1"], "pos": [100]})
        neg = pos.clone()

        with pytest.raises(ValueError, match="missing columns"):
            match_features(pos, neg, [], [], k=1)

    def test_scale_false_does_not_scale(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1"],
                "pos": [100],
                "ref": ["A"],
                "alt": ["T"],
                "cat": ["A"],
                "feat": [1000.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [200, 300],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat": ["A", "A"],
                "feat": [1001.0, 1002.0],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat"], k=1, scale=False)

        assert len(result) == 2
        assert "_scaled" not in str(result.columns)


def _apply_bin(values: list[float | None], edges: list[float], **kwargs) -> list[str]:
    df = pl.DataFrame({"x": values}, schema={"x": pl.Float64}).with_columns(
        bin_feature("x", edges, **kwargs).alias("bin")
    )
    return df["bin"].to_list()


class TestBinFeature:
    def test_returns_polars_expression(self) -> None:
        expr = bin_feature("x", [0, 1, 2])
        assert isinstance(expr, pl.Expr)

    def test_left_closed_basic(self) -> None:
        # Edges [0, 50, 100, 200, 500, 1000] -> 5 bins (b0..b4); left-closed.
        edges = [0, 50, 100, 200, 500, 1000]
        values = [0, 49, 50, 99, 100, 199, 200, 499, 500, 999, 1000]
        expected = ["b0", "b0", "b1", "b1", "b2", "b2", "b3", "b3", "b4", "b4", "b4"]
        assert _apply_bin(values, edges) == expected

    def test_left_closed_out_of_range(self) -> None:
        edges = [0, 50, 100]
        assert _apply_bin([-1, 100.1, 1000], edges) == [BIN_OOR, BIN_OOR, BIN_OOR]

    def test_right_closed_basic(self) -> None:
        # First bin is [lo, hi]; subsequent bins are (lo, hi].
        edges = [0, 0.001, 0.005, 0.5]
        values = [0, 0.0005, 0.001, 0.0011, 0.005, 0.0051, 0.5]
        expected = ["b0", "b0", "b0", "b1", "b1", "b2", "b2"]
        assert _apply_bin(values, edges, right_closed=True) == expected

    def test_right_closed_out_of_range(self) -> None:
        edges = [0, 0.001, 0.5]
        assert _apply_bin([-0.1, 0.6], edges, right_closed=True) == [BIN_OOR, BIN_OOR]

    def test_null_input_becomes_oor(self) -> None:
        edges = [0, 50, 100]
        assert _apply_bin([None, 50.0], edges) == [BIN_OOR, "b1"]

    def test_result_dtype_is_string(self) -> None:
        df = pl.DataFrame({"x": [1.0, 2.0]}).with_columns(
            bin_feature("x", [0, 1, 2]).alias("bin")
        )
        assert df.schema["bin"] == pl.String

    def test_assertion_on_too_few_edges(self) -> None:
        with pytest.raises(AssertionError):
            bin_feature("x", [0])

    def test_iter22_tss_dist_edges(self) -> None:
        # tss_proximal range is [0, 1000], so all valid values must bin in-range.
        values = [0, 49, 50, 99, 100, 199, 200, 499, 500, 999, 1000]
        bins = _apply_bin(values, TSS_DIST_BIN_EDGES)
        assert BIN_OOR not in bins

    def test_iter22_exon_dist_edges(self) -> None:
        # splicing exon_dist range after pre-filter is [0, 30].
        values = [0, 4, 5, 19, 20, 29, 30]
        bins = _apply_bin(values, EXON_DIST_BIN_EDGES)
        assert BIN_OOR not in bins
        assert bins == ["b0", "b0", "b1", "b1", "b2", "b2", "b2"]

    def test_iter24_maf_edges(self) -> None:
        # MAF range is [0, 0.5]. Right-closed convention: 0 and 0.5 must land
        # in real bins (not OOR), and the standard cutoffs (0.001, 0.005, 0.05)
        # land at the upper edge of their bin.
        bins = _apply_bin(
            [0.0, 0.0005, 0.001, 0.005, 0.05, 0.5],
            MAF_BIN_EDGES,
            right_closed=True,
        )
        assert BIN_OOR not in bins
        # 0 and 0.0005 share b0 (first bin is [0, 0.0005] inclusive).
        assert bins[0] == "b0"
        assert bins[0] == bins[1]
        # Upper edge of a bin lands in that bin (right-closed).
        # MAF=0.5 is the upper edge of the last bin.
        assert bins[-1] == "b19"


class TestMatchFeaturesAcceptsBinColumns:
    """End-to-end smoke test: bin columns work as exact-match categoricals."""

    def test_subset_conditional_na_matches_within_label_groups(self) -> None:
        # 2 positives + 2 negatives, all in chrom 1, two consequence groups.
        # bin column is "NA" for the non-matching subset, "b0" for the other.
        # match_features should respect the bin as a categorical.
        pos = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [100, 200],
                "ref": ["A", "C"],
                "alt": ["T", "G"],
                "consequence_group": ["distal", "tss_proximal"],
                "tss_dist": [10000.0, 50.0],
                "tss_dist_bin": ["NA", "b1"],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1", "1", "1"],
                "pos": [110, 210, 310, 410],
                "ref": ["A", "C", "A", "C"],
                "alt": ["T", "G", "T", "G"],
                "consequence_group": [
                    "distal",
                    "tss_proximal",
                    "distal",
                    "tss_proximal",
                ],
                "tss_dist": [11000.0, 60.0, 12000.0, 70.0],
                "tss_dist_bin": ["NA", "b1", "NA", "b1"],
            }
        )
        result = match_features(
            pos,
            neg,
            ["tss_dist"],
            ["chrom", "consequence_group", "tss_dist_bin"],
            k=1,
        )
        assert result.height == 4  # 2 positives + 2 matched negatives
        # Every match group has the same tss_dist_bin in pos and neg.
        groups = result.group_by(MATCH_GROUP_COL).agg(
            pl.col("tss_dist_bin").n_unique().alias("n")
        )
        assert (groups["n"] == 1).all()

    def test_passthrough_column_survives_matching(self) -> None:
        # Regression for issue #156: ld_score is dropped from the matching
        # call but kept in the output dataset as a passthrough column.
        pos = pl.DataFrame(
            {
                "chrom": ["1"],
                "pos": [100],
                "ref": ["A"],
                "alt": ["T"],
                "cat": ["x"],
                "MAF": [0.1],
                "ld_score": [42.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [200, 300],
                "ref": ["C", "G"],
                "alt": ["A", "T"],
                "cat": ["x", "x"],
                "MAF": [0.11, 0.5],
                "ld_score": [99.0, 7.0],
            }
        )
        result = match_features(pos, neg, ["MAF"], ["cat"], k=1)
        assert "ld_score" in result.columns
        assert result.height == 2


def test_bin_na_constant_distinct_from_bin_labels() -> None:
    # BIN_NA must not collide with any "b{i}" label that bin_feature emits, or
    # with the BIN_OOR sentinel, since both can co-occur in a single column.
    assert not BIN_NA.startswith("b") or not BIN_NA[1:].isdigit()
    assert BIN_NA != BIN_OOR


class TestAddSubsetDistanceBins:
    """Per-biotype distance-bin assignment helper (no longer called by the
    pipeline; kept for ad-hoc analysis)."""

    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "chrom": ["1"] * 4,
                "consequence_group": [
                    "tss_proximal",
                    "splicing",
                    "non_coding_transcript_exon_variant",
                    "distal",
                ],
                "distance_tss_pc": [25.0, 5000.0, 8000.0, 200000.0],
                "distance_tss_nc": [25.0, 5000.0, 600.0, 200000.0],
                "distance_exon_pc": [1000.0, 5.0, 5000.0, 50000.0],
                "distance_exon_nc": [1000.0, 5000.0, 5000.0, 50000.0],
            }
        )

    def test_only_tss_proximal_gets_tss_bins(self) -> None:
        out = add_subset_distance_bins(self._frame())
        rows = {r["consequence_group"]: r for r in out.to_dicts()}
        assert rows["tss_proximal"]["distance_tss_pc_bin"] == "b0"  # 25 ∈ [0, 50)
        assert rows["tss_proximal"]["distance_tss_nc_bin"] == "b0"
        for grp in ("splicing", "non_coding_transcript_exon_variant", "distal"):
            assert rows[grp]["distance_tss_pc_bin"] == BIN_NA
            assert rows[grp]["distance_tss_nc_bin"] == BIN_NA

    def test_only_splicing_gets_exon_bin(self) -> None:
        out = add_subset_distance_bins(self._frame())
        rows = {r["consequence_group"]: r for r in out.to_dicts()}
        # EXON_DIST_BIN_EDGES = [0, 5, 20, 30]; 5 ∈ [5, 20) → b1
        assert rows["splicing"]["distance_exon_pc_bin"] == "b1"
        for grp in ("tss_proximal", "non_coding_transcript_exon_variant", "distal"):
            assert rows[grp]["distance_exon_pc_bin"] == BIN_NA


class TestAddSubsetDistanceBinsV2:
    """Per-(subset, feature) distance binning with subset-prefixed labels."""

    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "consequence_group": [
                    "tss_proximal",
                    "splicing",
                    "distal",
                    "missense_variant",
                ],
                "distance_tss_pc": [50.0, 5000.0, 200000.0, 10000.0],
                "distance_exon_pc": [50.0, 3.0, 5000.0, 0.0],
            }
        )

    def test_single_feature_single_subset(self) -> None:
        out = add_subset_distance_bins_v2(
            self._frame(),
            {("tss_proximal", "distance_tss_pc"): [0, 100, 1000, float("inf")]},
        )
        labels = {
            r["consequence_group"]: r["distance_tss_pc_bin"] for r in out.to_dicts()
        }
        # tss_proximal: distance 50 → b0 (with subset prefix)
        assert labels["tss_proximal"] == "tss_prox:b0"
        # other subsets: BIN_NA
        for grp in ("splicing", "distal", "missense_variant"):
            assert labels[grp] == BIN_NA

    def test_shared_feature_across_subsets_gets_distinct_labels(self) -> None:
        out = add_subset_distance_bins_v2(
            self._frame(),
            {
                ("splicing", "distance_exon_pc"): [0, 5, 30, float("inf")],
                ("distal", "distance_exon_pc"): [0, 1000, 10000, float("inf")],
                ("tss_proximal", "distance_exon_pc"): [0, 100, 1000, float("inf")],
            },
        )
        labels = {
            r["consequence_group"]: r["distance_exon_pc_bin"] for r in out.to_dicts()
        }
        # tss_proximal: 50 ∈ [0, 100) → b0, with "tss_prox" prefix
        assert labels["tss_proximal"] == "tss_prox:b0"
        # splicing: 3 ∈ [0, 5) → b0, with "splicing" prefix
        assert labels["splicing"] == "splicing:b0"
        # distal: 5000 ∈ [1000, 10000) → b1, with "distal" prefix
        assert labels["distal"] == "distal:b1"
        # missense: not in scheme → BIN_NA
        assert labels["missense_variant"] == BIN_NA

    def test_multiple_features_create_multiple_columns(self) -> None:
        out = add_subset_distance_bins_v2(
            self._frame(),
            {
                ("tss_proximal", "distance_tss_pc"): [0, 100, 1000, float("inf")],
                ("tss_proximal", "distance_exon_pc"): [0, 100, 1000, float("inf")],
            },
        )
        # Both bin columns added.
        assert "distance_tss_pc_bin" in out.columns
        assert "distance_exon_pc_bin" in out.columns
        tss_row = out.filter(pl.col("consequence_group") == "tss_proximal").to_dicts()[
            0
        ]
        # tss_proximal: dist_tss_pc=50 → b0, dist_exon_pc=50 → b0
        assert tss_row["distance_tss_pc_bin"] == "tss_prox:b0"
        assert tss_row["distance_exon_pc_bin"] == "tss_prox:b0"

    def test_oor_label_for_value_above_max_edge(self) -> None:
        # distal/distance_exon_pc=5000 with edges [0, 100, 1000] would be OOR.
        # bin_feature returns "OOR" for in-range failure (> last edge).
        out = add_subset_distance_bins_v2(
            self._frame(),
            {("distal", "distance_exon_pc"): [0, 100, 1000]},
        )
        labels = {
            r["consequence_group"]: r["distance_exon_pc_bin"] for r in out.to_dicts()
        }
        # bin_feature default is left-closed; 5000 > 1000 → OOR
        assert labels["distal"] == "distal:OOR"


class TestAddTieredMafBin:
    """Per-subset MAF binning helper (no longer called by the pipeline;
    kept for ad-hoc analysis)."""

    def test_global_edges_only_assigns_bin_per_subset(self) -> None:
        df = pl.DataFrame(
            {
                "MAF": [0.001, 0.05, 0.001, 0.05, 0.5],
                "consequence_group": [
                    "missense_variant",
                    "missense_variant",  # 5bin range scheme
                    "distal",
                    "distal",  # 20bin scheme
                    "distal",
                ],
            }
        )
        scheme = {
            "missense_variant": MAF_BIN_EDGES_5,
            "distal": MAF_BIN_EDGES_20,
        }
        out = add_tiered_maf_bin(df, scheme)
        labels = out["MAF_bin"].to_list()
        # Different MAF within the same subset → different bin index.
        assert labels[0] != labels[1]
        # Different subset → different label prefix → cannot collide.
        assert labels[0] != labels[2]
        # Prefix encodes the subset name (truncated to 8 chars).
        assert all(":" in lab for lab in labels)
        assert all(lab.startswith("missense") for lab in labels[:2])
        assert all(lab.startswith("distal:") for lab in labels[2:])

    def test_unknown_consequence_group_gets_unknown_label(self) -> None:
        # MAF_BIN_EDGES_5 = [0, 0.001, 0.01, 0.05, 0.2, 0.5] right-closed:
        # bin 0=[0,0.001], 1=(0.001,0.01], 2=(0.01,0.05], 3=(0.05,0.2], 4=(0.2,0.5]
        df = pl.DataFrame(
            {"MAF": [0.001, 0.05], "consequence_group": ["foo", "missense_variant"]}
        )
        out = add_tiered_maf_bin(df, {"missense_variant": MAF_BIN_EDGES_5})
        assert out["MAF_bin"].to_list() == ["UNKNOWN", "missense:b2"]

    def test_null_maf_gets_oor_label(self) -> None:
        df = pl.DataFrame(
            {
                "MAF": [None, 0.05],
                "consequence_group": ["missense_variant", "missense_variant"],
            },
            schema={
                "MAF": pl.Float64,
                "consequence_group": pl.String,
            },
        )
        out = add_tiered_maf_bin(df, {"missense_variant": MAF_BIN_EDGES_5})
        labels = out["MAF_bin"].to_list()
        assert labels[0] == "missense:OOR"
        assert labels[1].startswith("missense:b")

    def test_empty_dataframe_through_helper(self) -> None:
        df = pl.DataFrame(
            schema={
                "MAF": pl.Float64,
                "consequence_group": pl.String,
            }
        )
        out = add_tiered_maf_bin(df, {"distal": MAF_BIN_EDGES_5})
        assert out.height == 0
        assert "MAF_bin" in out.columns

    def test_v1_scheme_has_expected_subsets(self) -> None:
        expected = {
            "distal",
            "tss_proximal",
            "non_coding_transcript_exon_variant",
            "3_prime_UTR_variant",
            "5_prime_UTR_variant",
            "missense_variant",
            "synonymous_variant",
            "splicing",
            "mature_miRNA_variant",
            "stop_retained_variant",
            "coding_sequence_variant",
        }
        assert set(MAF_TIERED_V1.keys()) == expected
