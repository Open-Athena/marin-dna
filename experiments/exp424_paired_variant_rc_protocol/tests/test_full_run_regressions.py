from __future__ import annotations

import polars as pl

from analyze import AGGREGATE_VIEWS, choose_aggregate_view


def test_validation_sort_selects_one_scalar_without_consulting_test() -> None:
    summary = pl.DataFrame(
        {
            "view": [*AGGREGATE_VIEWS, "forward_signed"],
            "validation_mean_ap": [0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
            "test_mean_ap": [0.9, 0.8, 0.7, 0.6, 0.1, 1.0],
        }
    )
    assert choose_aggregate_view(summary) == "max_abs"
