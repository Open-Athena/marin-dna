from __future__ import annotations

import numpy as np
import polars as pl

from inspect_candidates import frequency_rows, response_orientation


def test_response_orientation_tracks_strand_and_position() -> None:
    forward = np.array([[0, 1, 2], [0, 8, 1]], dtype=np.float32)
    reverse = np.array([[0, 5, 1], [0, 2, 3]], dtype=np.float32)

    focal_reverse, focal_position = response_orientation(forward, reverse, "focal")
    np.testing.assert_array_equal(focal_reverse, [True, False])
    np.testing.assert_array_equal(focal_position, [1, 1])

    max_reverse, max_position = response_orientation(forward, reverse, "local_max")
    np.testing.assert_array_equal(max_reverse, [True, False])
    np.testing.assert_array_equal(max_position, [1, 1])


def test_frequency_rows_separates_top_and_remainder() -> None:
    frame = pl.DataFrame(
        {
            "class": ["x", "x"],
            "feature_id": [7, 7],
            "is_top": [True, False],
            "ref_context": ["A" * 31, "C" * 31],
            "alt_context": ["A" * 15 + "G" + "A" * 15, "C" * 31],
        }
    )

    frequencies = pl.DataFrame(frequency_rows(frame))
    focal = frequencies.filter(
        (pl.col("relative_position") == 0) & (pl.col("allele") == "alt")
    )
    assert (
        focal.filter((pl.col("subset") == "top") & (pl.col("base") == "G"))[
            "frequency"
        ].item()
        == 1
    )
    assert (
        focal.filter((pl.col("subset") == "remainder") & (pl.col("base") == "C"))[
            "frequency"
        ].item()
        == 1
    )
