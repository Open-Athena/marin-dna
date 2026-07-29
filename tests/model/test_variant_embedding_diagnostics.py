"""Tests for mutation-class embedding diagnostics."""

import numpy as np
import pytest

from marin_dna.model.variant_embedding_diagnostics import (
    genomic_position_regions,
    neighbor_locality_summary,
    residualize_features_by_category,
    snv_substitution_classes,
)


def test_snv_substitution_classes_pair_reverse_complements():
    directed, canonical = snv_substitution_classes(
        ["A", "T", "C", "G"],
        ["C", "G", "T", "A"],
    )
    np.testing.assert_array_equal(directed, ["A>C", "T>G", "C>T", "G>A"])
    np.testing.assert_array_equal(canonical, ["A>C", "A>C", "C>T", "C>T"])


@pytest.mark.parametrize(
    ("ref", "alt", "message"),
    [
        (["A"], ["A"], "must differ"),
        (["N"], ["A"], "reference"),
        (["A"], ["N"], "alternate"),
        (["A", "C"], ["T"], "shape mismatch"),
        ([], [], "at least one"),
    ],
)
def test_snv_substitution_classes_reject_invalid_inputs(ref, alt, message):
    with pytest.raises(AssertionError, match=message):
        snv_substitution_classes(ref, alt)


def test_residualize_features_by_category_matches_one_hot_regression():
    features = np.array(
        [[0.0, 0.0], [2.0, 0.0], [10.0, 2.0], [12.0, 2.0]],
        dtype=np.float16,
    )
    categories = np.array(["a", "a", "b", "b"])

    residuals, explained_fraction = residualize_features_by_category(
        features,
        categories,
    )

    assert residuals.dtype == np.float32
    np.testing.assert_array_equal(
        residuals,
        np.array([[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]),
    )
    assert explained_fraction == pytest.approx(104.0 / 108.0)
    for category in np.unique(categories):
        np.testing.assert_allclose(
            residuals[categories == category].mean(axis=0),
            0.0,
            atol=1e-7,
        )


@pytest.mark.parametrize(
    ("features", "categories", "message"),
    [
        (np.zeros((2, 2, 1)), ["a", "b"], "feature matrix"),
        (np.zeros((2, 2)), ["a"], "category count"),
        (np.zeros((2, 2)), ["a", ""], "non-empty"),
        (np.array([[0.0, np.nan], [1.0, 2.0]]), ["a", "b"], "non-finite"),
        (np.ones((2, 2)), ["a", "b"], "constant"),
    ],
)
def test_residualize_features_by_category_rejects_invalid_inputs(
    features,
    categories,
    message,
):
    with pytest.raises(AssertionError, match=message):
        residualize_features_by_category(features, categories)


def test_genomic_position_regions_handles_unsorted_duplicates():
    labels = genomic_position_regions(
        np.array([1_000, 10, 11, 1_000, 300, 12]),
        max_gap=10,
    )
    np.testing.assert_array_equal(
        labels,
        ["region 03", "region 01", "region 01", "region 03", "region 02", "region 01"],
    )


@pytest.mark.parametrize(
    ("positions", "max_gap", "message"),
    [
        (np.array([[1, 2]]), 10, "one-dimensional"),
        (np.array([], dtype=np.int64), 10, "at least one"),
        (np.array([1.0, 2.0]), 10, "integers"),
        (np.array([-1, 2]), 10, "non-negative"),
        (np.array([1, 2]), -1, "non-negative"),
    ],
)
def test_genomic_position_regions_rejects_invalid_inputs(
    positions,
    max_gap,
    message,
):
    with pytest.raises(AssertionError, match=message):
        genomic_position_regions(positions, max_gap=max_gap)


def test_neighbor_locality_summary_uses_nonself_nearest_neighbors():
    summary = neighbor_locality_summary(
        np.array([[0.0], [0.1], [1.0], [10.0]]),
        np.array([10, 10, 100, 1_000]),
        np.array(["region 01", "region 01", "region 01", "region 02"]),
        n_neighbors=1,
        context_size=255,
    )
    assert summary == {
        "same exact position fraction": 0.5,
        "same assayed region fraction": 0.75,
        "within 127 bp fraction": 0.75,
        "within 255 bp fraction": 0.75,
        "median absolute position distance (bp)": 45.0,
    }
