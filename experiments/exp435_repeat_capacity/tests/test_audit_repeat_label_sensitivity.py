from __future__ import annotations

import numpy as np
import polars as pl

from analyze_repeat_label_sensitivity import add_stratum_calls
from association_common import bh_adjust
from audit_repeat_label_sensitivity import validate_family


def test_audit_recomputes_bh_and_strict_calls() -> None:
    welch_p = np.asarray([0.01, 0.20])
    mann_p = np.asarray([0.02, 0.30])
    welch_q = bh_adjust(welch_p)
    mann_q = bh_adjust(mann_p)
    base = pl.DataFrame(
        {
            "arm": ["block19-25m", "block19-25m"],
            "block": [19, 19],
            "budget": [25_000_200, 25_000_200],
            "orientation": ["forward", "forward"],
            "pooling": ["focal", "focal"],
            "response": ["delta", "delta"],
            "response_role": ["primary", "primary"],
            "target_kind": ["overall", "overall"],
            "target": ["overall", "overall"],
            "feature_id": [1, 2],
            "mean_difference": [1.0, -1.0],
            "rank_biserial": [0.1, -0.1],
            "welch_p": welch_p,
            "mann_whitney_p": mann_p,
            "welch_q": welch_q,
            "mann_whitney_q": mann_q,
            "minimum_q": np.fmin(welch_q, mann_q),
        }
    )
    archived = add_stratum_calls(base, "repeat_free_window")
    assert validate_family(archived, "repeat_free_window").equals(archived)
