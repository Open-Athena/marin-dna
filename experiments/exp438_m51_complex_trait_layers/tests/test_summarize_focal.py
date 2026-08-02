from __future__ import annotations

import polars as pl

from summarize_focal import layer_winners, target_summary


def test_target_summary_and_layer_winner() -> None:
    frame = pl.DataFrame(
        {
            "arm": ["block01-25m", "block01-25m", "block10-25m", "block10-25m"],
            "block": [1, 1, 10, 10],
            "orientation": ["forward"] * 4,
            "response": ["abs_delta"] * 4,
            "target_kind": ["overall"] * 4,
            "target": ["overall"] * 4,
            "inferential": [True] * 4,
            "welch_q": [0.01, 0.2, 0.01, 0.02],
            "mann_whitney_q": [0.02, 0.3, 0.01, 0.03],
            "best_auprc": [0.2, 0.15, 0.25, 0.22],
            "prevalence": [0.1] * 4,
        }
    )
    summary = target_summary(frame)
    assert summary.height == 2
    block10 = summary.filter(pl.col("block") == 10).row(0, named=True)
    assert block10["both_test_discoveries"] == 2
    assert block10["best_auprc_lift"] == 2.5
    winner = layer_winners(summary).row(0, named=True)
    assert winner["winning_block"] == 10
