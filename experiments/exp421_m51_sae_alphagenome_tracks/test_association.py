from __future__ import annotations

import numpy as np

from association import (
    benjamini_hochberg,
    group_bootstrap_correlation,
    group_center,
    pearson,
    pearson_pvalue,
    spearman,
)


def test_pearson_and_spearman_capture_different_relationships() -> None:
    x = np.arange(1, 8, dtype=np.float64)
    y = x**3

    assert 0.8 < pearson(x, y) < 1.0
    rho, pvalue = spearman(x, y)
    assert rho == 1.0
    assert pvalue < 0.001


def test_spearman_maps_constant_input_to_null() -> None:
    constant = np.ones(5)
    varying = np.arange(5, dtype=np.float64)

    assert pearson_pvalue(constant, varying) == 1.0
    rho, pvalue = spearman(constant, varying)
    assert rho == 0.0
    assert pvalue == 1.0


def test_group_center_removes_group_means() -> None:
    groups = np.array([0, 0, 1, 1])
    centered = group_center(np.array([1.0, 3.0, 10.0, 14.0]), groups)

    np.testing.assert_allclose(centered, [-1.0, 1.0, -2.0, 2.0])
    np.testing.assert_allclose(
        [centered[groups == group].mean() for group in np.unique(groups)],
        0.0,
        atol=1e-12,
    )


def test_group_bootstrap_is_deterministic_and_contains_estimate() -> None:
    groups = np.repeat(np.arange(12), 2)
    x = np.arange(len(groups), dtype=np.float64)
    y = x + np.tile([-0.2, 0.2], 12)

    first = group_bootstrap_correlation(x, y, groups, random_seed=7, samples=200)
    second = group_bootstrap_correlation(x, y, groups, random_seed=7, samples=200)

    assert first == second
    assert first[0] <= pearson(x, y) <= first[1]


def test_benjamini_hochberg_preserves_input_order() -> None:
    qvalues = benjamini_hochberg(np.array([0.04, 0.001, 0.03, 0.8]))

    np.testing.assert_allclose(
        qvalues, [0.05333333333333334, 0.004, 0.05333333333333334, 0.8]
    )
