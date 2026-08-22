from __future__ import annotations

import numpy as np
import pandas as pd

from exp479_mntp.dual_mode_synthesis import (
    nucleotide_readouts,
    paired_nucleotide_comparison,
    symmetric_two_pass_log_probs,
    vep_decomposition_scores,
)


def test_symmetric_distribution_is_normalized_and_direction_invariant() -> None:
    left = np.log(np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]]))
    right = np.log(np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]))
    prior = np.log(np.array([0.3, 0.2, 0.2, 0.3]))
    first = symmetric_two_pass_log_probs(left, right, log_prior=prior)
    second = symmetric_two_pass_log_probs(right, left, log_prior=prior)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.exp(first).sum(axis=1), 1)


def test_nucleotide_readout_and_paired_comparison_use_identical_ids() -> None:
    directional = pd.DataFrame(
        {
            "sample_id": [10, 11, 12, 13],
            "target_nucleotide_class": [0, 1, 2, 3],
            **{f"left_logp_{base}": np.log([0.25, 0.25, 0.25, 0.25]) for base in "acgt"},
            **{
                f"right_logp_{base}": np.log(
                    [0.7 if index == target else 0.1 for target in range(4)]
                )
                for index, base in enumerate("acgt")
            },
        }
    )
    scores = nucleotide_readouts(directional, prior=np.full(4, 0.25), alpha=1)
    comparison = paired_nucleotide_comparison(
        scores,
        candidate="two_pass_symmetric",
        baseline="source_left",
        n_bootstrap=100,
    )
    assert comparison["n_targets"] == 4
    assert comparison["nucleotide_ce_delta"] < 0
    assert comparison["nucleotide_accuracy_delta"] > 0


def test_vep_decomposition_reconstructs_source_and_replaces_only_central_term() -> None:
    frame = pd.DataFrame(
        {
            "source_clm_avg": [3.0, -1.0],
            "source_conditional_avg": [1.0, 0.5],
            "source_two_pass_symmetric": [1.5, 0.25],
        }
    )
    result = vep_decomposition_scores(frame)
    np.testing.assert_allclose(
        result["masked_central"] + result["context_residual"],
        result["source_clm_avg"],
    )
    np.testing.assert_allclose(result["two_pass_plus_residual"], [3.5, -1.25])
