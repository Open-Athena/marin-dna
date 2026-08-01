import polars as pl

from analyze_heldout_perturbations import splice_context_contrasts


def test_splice_context_contrast_uses_frozen_positions() -> None:
    rows = []
    for source_panel_row in range(64):
        for relative_position in range(-12, 5):
            for alternate_state in "CGT":
                rows.append(
                    {
                        "response_role": "primary",
                        "perturbation_type": "splice_saturation",
                        "class": "splice_acceptor_variant",
                        "analysis_feature_id": 11698,
                        "context_group": "untouched_test_hash",
                        "source_panel_row": source_panel_row,
                        "relative_position": relative_position,
                        "alternate_state": alternate_state,
                        "response_score": (
                            3.0 if relative_position in {-1, 0} else 1.0
                        ),
                    }
                )
    contexts, summary = splice_context_contrasts(pl.DataFrame(rows))
    assert contexts.height == 64
    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["target_positions"] == "-1,0"
    assert row["mean_target_response"] == 3.0
    assert row["mean_other_response"] == 1.0
    assert row["mean_target_minus_other"] == 2.0
    assert row["bootstrap_ci95_low"] == 2.0
    assert row["bootstrap_ci95_high"] == 2.0
