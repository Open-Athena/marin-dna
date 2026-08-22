from __future__ import annotations

import numpy as np
import pandas as pd

from exp479_mntp.two_pass_vep import (
    _normal_macro,
    log_probability_ratios,
    primary_paired_comparison,
    two_pass_log_probabilities,
)


def test_symmetric_two_pass_is_invariant_to_direction_swap() -> None:
    left = np.log(np.array([[0.55, 0.20, 0.15, 0.10], [0.10, 0.20, 0.30, 0.40]]))
    right = np.log(np.array([[0.20, 0.50, 0.20, 0.10], [0.40, 0.30, 0.20, 0.10]]))
    prior = np.log(np.array([0.3, 0.2, 0.2, 0.3]))
    forward = two_pass_log_probabilities(left, right, log_prior=prior)
    swapped = two_pass_log_probabilities(right, left, log_prior=prior)
    np.testing.assert_allclose(
        forward["source_two_pass_symmetric"],
        swapped["source_two_pass_symmetric"],
    )
    np.testing.assert_allclose(
        np.exp(forward["source_two_pass_symmetric"]).sum(axis=1),
        1,
    )
    assert np.array_equal(forward["source_conditional_left"], left)
    assert np.array_equal(forward["source_conditional_right"], right)


def test_symmetric_alpha_zero_reduces_to_directional_log_mean() -> None:
    left = np.log(np.array([[0.6, 0.2, 0.1, 0.1]]))
    right = np.log(np.array([[0.1, 0.2, 0.3, 0.4]]))
    result = two_pass_log_probabilities(
        left,
        right,
        log_prior=np.log(np.full(4, 0.25)),
        symmetric_alpha=0,
    )
    np.testing.assert_allclose(
        result["source_two_pass_symmetric"],
        result["source_conditional_avg"],
    )


def test_log_probability_ratio_uses_alternate_minus_reference() -> None:
    log_probabilities = np.log(np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]]))
    result = log_probability_ratios(
        log_probabilities,
        reference_indices=np.array([0, 3]),
        alternate_indices=np.array([1, 2]),
    )
    np.testing.assert_allclose(result, np.log([0.3 / 0.5, 0.3 / 0.4]))


def test_normal_macro_combines_child_delta_uncertainty() -> None:
    result = _normal_macro(
        [
            {"delta": 0.1, "se": 0.02, "n_rows": 10},
            {"delta": 0.2, "se": 0.04, "n_rows": 20},
        ]
    )
    assert np.isclose(result["delta"], 0.15)
    assert np.isclose(result["se"], np.sqrt(0.02**2 + 0.04**2) / 2)
    assert result["n_groups"] == 2
    assert result["n_rows"] == 30


def test_complex_primary_comparison_uses_paired_match_groups() -> None:
    labels = np.tile([0, 1], 40)
    variants = pd.DataFrame(
        {
            "label": labels,
            "match_group": np.repeat(np.arange(40), 2),
        }
    )
    scores = pd.DataFrame(
        {
            "candidate": labels.astype(float),
            "baseline": 1 - labels.astype(float),
        }
    )
    result = primary_paired_comparison(
        "complex_traits",
        variants,
        scores,
        candidate="candidate",
        baseline="baseline",
        n_bootstrap=100,
    )
    assert result["dataset"] == "complex_traits"
    assert result["delta"] > 0
    assert result["ci_low"] > 0
