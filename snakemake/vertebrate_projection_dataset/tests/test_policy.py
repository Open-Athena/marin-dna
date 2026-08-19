from __future__ import annotations

import polars as pl
import pytest
from marin_dna_vertebrate_projection.policy import (
    FULL_WINDOW_POLICY,
    build_projection_requests,
    centered_landmark_policy,
)


def _anchors() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": ["a1", "a2"],
            "source_chrom": ["chr1", "chr2"],
            "source_start": [100, 1_000],
            "source_end": [355, 1_255],
            "region_label": ["cds", "ccre_non_promoter"],
        }
    )


def test_full_window_policy_preserves_exact_baseline_intervals_and_gate() -> None:
    policy = FULL_WINDOW_POLICY
    requests = build_projection_requests(_anchors(), policy)

    assert policy.pre_resize_min_length == 128
    assert policy.pre_resize_max_length == 512
    assert requests.select("projection_start", "projection_end").rows() == [
        (100, 355),
        (1_000, 1_255),
    ]


@pytest.mark.parametrize(
    ("width", "expected_interval", "expected_gate"),
    [
        (1, (227, 228), (1, 2)),
        (17, (219, 236), (9, 34)),
        (33, (211, 244), (17, 66)),
        (65, (195, 260), (33, 130)),
        (129, (163, 292), (65, 258)),
    ],
)
def test_centered_landmark_policies_use_one_exact_central_nucleotide(
    width: int,
    expected_interval: tuple[int, int],
    expected_gate: tuple[int, int],
) -> None:
    policy = centered_landmark_policy(width)
    request = build_projection_requests(_anchors().head(1), policy).row(0, named=True)

    assert (request["projection_start"], request["projection_end"]) == expected_interval
    assert request["projection_start"] <= 227 < request["projection_end"]
    assert (policy.pre_resize_min_length, policy.pre_resize_max_length) == expected_gate


@pytest.mark.parametrize("width", [0, 2])
def test_centered_landmark_policy_rejects_nonpositive_or_even_width(width: int) -> None:
    with pytest.raises(AssertionError, match="positive odd"):
        centered_landmark_policy(width)


def test_centered_landmark_policy_rejects_width_larger_than_anchor() -> None:
    with pytest.raises(AssertionError, match="landmark cannot be wider"):
        build_projection_requests(_anchors(), centered_landmark_policy(257))
