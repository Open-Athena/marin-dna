"""Mutation-class diagnostics for allele-pair embedding features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_NUCLEOTIDES = frozenset("ACGT")
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def snv_substitution_classes(
    ref: Sequence[str] | np.ndarray,
    alt: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed and reverse-complement-canonical SNV class labels.

    Directed labels retain all 12 possible substitutions (for example,
    ``A>C``). The canonical label identifies a substitution with its
    reverse-complement partner by taking the lexicographically smaller of
    ``REF>ALT`` and ``complement(REF)>complement(ALT)``.
    """
    ref_array = np.asarray(ref, dtype=str)
    alt_array = np.asarray(alt, dtype=str)
    assert ref_array.ndim == 1, f"expected one-dimensional refs, got {ref_array.shape}"
    assert alt_array.ndim == 1, f"expected one-dimensional alts, got {alt_array.shape}"
    assert ref_array.shape == alt_array.shape, (
        f"ref/alt shape mismatch: {ref_array.shape} vs {alt_array.shape}"
    )
    assert len(ref_array) > 0, "need at least one SNV"
    assert set(ref_array) <= _NUCLEOTIDES, (
        f"invalid reference alleles: {sorted(set(ref_array) - _NUCLEOTIDES)}"
    )
    assert set(alt_array) <= _NUCLEOTIDES, (
        f"invalid alternate alleles: {sorted(set(alt_array) - _NUCLEOTIDES)}"
    )
    assert np.all(ref_array != alt_array), "SNV reference and alternate must differ"

    directed = np.asarray(
        [f"{ref_base}>{alt_base}" for ref_base, alt_base in zip(ref_array, alt_array)],
        dtype=str,
    )
    reverse_complement = np.asarray(
        [
            f"{_COMPLEMENT[ref_base]}>{_COMPLEMENT[alt_base]}"
            for ref_base, alt_base in zip(ref_array, alt_array)
        ],
        dtype=str,
    )
    canonical = np.asarray(
        [
            min(directed_label, reverse_complement_label)
            for directed_label, reverse_complement_label in zip(
                directed,
                reverse_complement,
            )
        ],
        dtype=str,
    )
    assert len(set(directed)) <= 12
    assert len(set(canonical)) <= 6
    return directed, canonical


def residualize_features_by_category(
    features: np.ndarray,
    categories: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Subtract per-category feature means and return categorical R-squared.

    This is the residual from independently regressing every feature dimension
    on an intercept plus a one-hot encoding of ``categories``. The returned
    scalar is the fraction of total centered sum of squares explained jointly
    by those category means across all feature dimensions.
    """
    feature_array = np.asarray(features, dtype=np.float32)
    category_array = np.asarray(categories, dtype=str)
    assert feature_array.ndim == 2, (
        f"expected feature matrix [N,D], got {feature_array.shape}"
    )
    assert feature_array.shape[0] > 1 and feature_array.shape[1] > 0
    assert np.isfinite(feature_array).all(), "features contain non-finite values"
    assert category_array.ndim == 1, (
        f"expected one-dimensional categories, got {category_array.shape}"
    )
    assert len(category_array) == len(feature_array), (
        f"category count {len(category_array)} != row count {len(feature_array)}"
    )
    assert np.all(category_array != ""), "category labels must be non-empty"

    fitted = np.empty_like(feature_array)
    unique_categories = np.unique(category_array)
    assert len(unique_categories) >= 1
    for category in unique_categories:
        mask = category_array == category
        assert mask.any()
        fitted[mask] = feature_array[mask].mean(axis=0, dtype=np.float64)

    residuals = feature_array - fitted
    assert residuals.dtype == np.float32
    assert np.isfinite(residuals).all()
    for category in unique_categories:
        category_mean = residuals[category_array == category].mean(
            axis=0,
            dtype=np.float64,
        )
        np.testing.assert_allclose(category_mean, 0.0, atol=1e-5, rtol=0.0)

    centered = feature_array.astype(np.float64) - feature_array.mean(
        axis=0,
        dtype=np.float64,
    )
    total_sum_squares = float(np.square(centered).sum())
    assert total_sum_squares > 0.0, "cannot residualize a constant feature matrix"
    residual_sum_squares = float(np.square(residuals.astype(np.float64)).sum())
    explained_fraction = 1.0 - residual_sum_squares / total_sum_squares
    assert -1e-6 <= explained_fraction <= 1.0 + 1e-6
    explained_fraction = float(np.clip(explained_fraction, 0.0, 1.0))
    return residuals, explained_fraction
