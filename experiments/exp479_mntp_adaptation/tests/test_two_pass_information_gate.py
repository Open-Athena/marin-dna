from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from exp479_mntp.two_pass_information_gate import (
    combine_directional_log_probs,
    reverse_complement_token_ids,
    select_alpha,
)


def test_reverse_complement_tokens_is_an_involution_and_preserves_unknowns() -> None:
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 1] + [1] * 250])
    reversed_once = reverse_complement_token_ids(
        input_ids,
        canonical_token_ids=(3, 4, 5, 6),
        bos_token_id=2,
    )
    assert reversed_once[0, :6].tolist() == [2, 1, 1, 1, 1, 1]
    assert reversed_once[0, -5:].tolist() == [1, 3, 4, 5, 6]
    reversed_twice = reverse_complement_token_ids(
        reversed_once,
        canonical_token_ids=(3, 4, 5, 6),
        bos_token_id=2,
    )
    assert torch.equal(reversed_twice, input_ids)


def test_reverse_target_position_mapping_is_an_involution() -> None:
    positions = torch.tensor([0, 1, 127, 253, 254])
    reverse_positions = 254 - positions
    assert reverse_positions.tolist() == [254, 253, 127, 1, 0]
    assert torch.equal(254 - reverse_positions, positions)


def test_alpha_zero_is_bit_exact_to_left_distribution() -> None:
    left = np.log(np.array([[0.5, 0.2, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]))
    right = np.log(np.array([[0.1, 0.2, 0.2, 0.5], [0.4, 0.3, 0.2, 0.1]]))
    combined = combine_directional_log_probs(
        left,
        right,
        log_prior=np.log(np.full(4, 0.25)),
        alpha=0,
    )
    assert np.array_equal(combined, left)


def test_prior_corrected_product_realigned_right_can_improve_calibration() -> None:
    rows: list[dict[str, float | int]] = []
    for sample_id, target in enumerate((0, 0, 1, 1, 2, 2, 3, 3)):
        left = np.log(np.full(4, 0.25))
        right_prob = np.full(4, 0.1)
        right_prob[target] = 0.7
        right = np.log(right_prob)
        row: dict[str, float | int] = {
            "sample_id": sample_id,
            "target_nucleotide_class": target,
        }
        for index, base in enumerate("acgt"):
            row[f"left_logp_{base}"] = left[index]
            row[f"right_logp_{base}"] = right[index]
        rows.append(row)
    alpha, curve = select_alpha(pd.DataFrame(rows), log_prior=np.log(np.full(4, 0.25)))
    assert alpha == 1
    assert curve.iloc[-1]["nucleotide_ce"] < curve.iloc[0]["nucleotide_ce"]
    assert curve.iloc[-1]["accuracy"] == 1


def test_combine_rejects_misaligned_shapes_or_alpha() -> None:
    valid = np.log(np.full((2, 4), 0.25))
    for right, prior, alpha in (
        (valid[:, :3], np.log(np.full(4, 0.25)), 0.5),
        (valid, np.log(np.full(3, 1 / 3)), 0.5),
        (valid, np.log(np.full(4, 0.25)), 1.1),
    ):
        try:
            combine_directional_log_probs(valid, right, log_prior=prior, alpha=alpha)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid two-pass combination was accepted")
