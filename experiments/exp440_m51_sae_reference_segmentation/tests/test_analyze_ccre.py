from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from analyze_ccre import (
    annotate_focal_ccre,
    contrast_codes,
    orientation_concordant_hits,
)


def test_annotate_focal_ccre_uses_half_open_focal_coordinate() -> None:
    panel = pl.DataFrame(
        {
            "panel_row": [0, 1, 2, 3],
            "chrom": ["1", "1", "1", "2"],
            "start": [0, 173, 223, 0],
            "end": [255, 428, 478, 255],
        }
    )
    ccre = pl.DataFrame(
        {
            "chrom": ["1", "1", "2"],
            "start": [100, 300, 127],
            "end": [200, 350, 128],
            "cre_class": ["pELS", "PLS", "dELS"],
        }
    )
    observed = annotate_focal_ccre(panel, ccre)
    assert observed["focal_position0"].to_list() == [127, 300, 350, 127]
    assert observed["ccre_subtype"].to_list() == ["pELS", "PLS", None, "dELS"]
    assert observed["ccre_group"].to_list() == [
        "els",
        "pls",
        "other_or_none",
        "els",
    ]
    np.testing.assert_array_equal(contrast_codes(observed, "els"), [1, 0, 0, 1])
    np.testing.assert_array_equal(contrast_codes(observed, "pls"), [0, 1, 0, 0])


def test_annotate_focal_ccre_rejects_overlapping_intervals() -> None:
    panel = pl.DataFrame(
        {"panel_row": [0], "chrom": ["1"], "start": [0], "end": [255]}
    )
    ccre = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [100, 150],
            "end": [200, 250],
            "cre_class": ["pELS", "dELS"],
        }
    )
    with pytest.raises(AssertionError):
        annotate_focal_ccre(panel, ccre)


def test_orientation_concordant_hits_requires_matching_primary_sign() -> None:
    rows = []
    for orientation, feature_id, effect, primary in (
        ("forward", 3, 0.4, True),
        ("reverse_complement", 3, 0.3, True),
        ("forward", 4, 0.5, True),
        ("reverse_complement", 4, -0.4, True),
        ("forward", 5, 0.6, True),
        ("reverse_complement", 5, 0.5, False),
    ):
        rows.append(
            {
                "block": 10,
                "arm": "block10-25m",
                "ccre_contrast": "els",
                "feature_id": feature_id,
                "orientation": orientation,
                "rank_biserial": effect,
                "mean_difference": effect,
                "welch_q": 0.001,
                "mwu_q": 0.001,
                "auprc": 0.5,
                "primary_association": primary,
            }
        )
    observed = orientation_concordant_hits(pl.DataFrame(rows))
    assert observed["feature_id"].to_list() == [3]
    assert observed["minimum_abs_rank_biserial"].to_list() == [0.3]
    assert observed["effect_direction"].to_list() == [1]
