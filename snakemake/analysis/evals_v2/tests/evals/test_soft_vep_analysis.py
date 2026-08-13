"""Pure analysis-harness tests; no S3 access."""

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.soft_vep_analysis import (
    ARMS,
    NON_DISTAL_SUBSETS,
    SPECIALIST_ARM,
    SYNCHRONIZED_STEPS,
    add_llr_scores,
    compute_rank_agreement,
    confidence_filtered_rank_reversals,
    earliest_persistent_specialist_wins,
    exp232_manifest,
    permute_labels_within_groups,
    plot_exp232_specialist_auprc_vs_brier,
    validate_aligned_bundles,
)
from marin_dna_evals.soft_vep_metrics import (
    AUPRC,
    CALIBRATED_BRIER,
    MEAN_GAP_GLOBAL,
)


def test_exp232_manifest_is_exact_inventory():
    manifest = exp232_manifest()
    assert len(manifest) == 48
    assert SYNCHRONIZED_STEPS == (500, 1000, 1500, 2000, 3000, 3500, 4000, 4500, 4999)
    assert manifest[manifest["step"] == 2500]["arm"].tolist() == ["bg", "cds", "utr3"]
    assert manifest["uri"].is_unique
    assert manifest["split"].eq("train").all()


def test_add_llr_scores_uses_signed_fwd_rc_average():
    frame = pd.DataFrame(
        {
            "label": [True, False],
            "subset": ["x", "x"],
            "match_group": [0, 0],
            "llr_fwd": [-2.0, 1.0],
            "llr_rc": [-4.0, 3.0],
        }
    )
    result = add_llr_scores(frame)
    assert result["minus_llr_avg"].tolist() == pytest.approx([3.0, -2.0])
    assert result["minus_llr_fwd"].tolist() == pytest.approx([2.0, -1.0])


def test_validate_aligned_bundles_fails_on_label_drift():
    base = pd.DataFrame(
        {
            "label": [True, False],
            "subset": ["x", "x"],
            "match_group": [0, 0],
        }
    )
    changed = base.copy()
    changed.loc[0, "label"] = False
    with pytest.raises(AssertionError, match="joint model bootstrap"):
        validate_aligned_bundles({"a": base, "b": changed})


def test_label_permutation_preserves_one_positive_per_group():
    groups = pd.Series([0, 0, 0, 1, 1, 1, 1])
    labels = permute_labels_within_groups(groups, rng=459)
    assert labels.groupby(groups).agg(["sum", "size"]).to_dict("index") == {
        0: {"sum": 1, "size": 3},
        1: {"sum": 1, "size": 4},
    }


def _synthetic_point_metrics() -> pd.DataFrame:
    rows = []
    for subset in NON_DISTAL_SUBSETS:
        specialist = SPECIALIST_ARM[subset]
        for step in SYNCHRONIZED_STEPS:
            for arm_index, arm in enumerate(ARMS):
                specialist_bonus = 0.0
                if arm == specialist:
                    specialist_bonus = 10.0 if step >= 1000 else -10.0
                auprc = arm_index + specialist_bonus
                rows.extend(
                    [
                        {
                            "arm": arm,
                            "step": step,
                            "subset": subset,
                            "metric": AUPRC,
                            "value": auprc,
                        },
                        {
                            "arm": arm,
                            "step": step,
                            "subset": subset,
                            "metric": MEAN_GAP_GLOBAL,
                            "value": auprc,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_rank_agreement_exact_match_has_no_reversals():
    summary, pairs = compute_rank_agreement(_synthetic_point_metrics())
    matched = summary[
        (summary["metric"] == MEAN_GAP_GLOBAL)
        & (summary["reference"] == "same_step_auprc")
    ]
    assert np.allclose(matched["spearman"], 1.0)
    assert np.allclose(matched["kendall"], 1.0)
    assert matched["pairwise_reversals"].eq(0).all()
    matched_pairs = pairs[
        (pairs["metric"] == MEAN_GAP_GLOBAL) & (pairs["reference"] == "same_step_auprc")
    ]
    assert not matched_pairs["reversal"].any()


def test_earliest_persistent_specialist_win_uses_two_stored_steps():
    wins = earliest_persistent_specialist_wins(_synthetic_point_metrics())
    assert wins["earliest_persistent_step"].eq(1000).all()
    assert wins["final_step_win"].all()


def test_confidence_filter_drops_unresolved_pairs():
    pairwise = pd.DataFrame(
        [
            {
                "step": 500,
                "subset": "s",
                "metric": AUPRC,
                "arm_a": "a",
                "arm_b": "b",
                "ci_low": 0.1,
                "ci_high": 0.3,
            },
            {
                "step": 500,
                "subset": "s",
                "metric": MEAN_GAP_GLOBAL,
                "arm_a": "a",
                "arm_b": "b",
                "ci_low": -0.2,
                "ci_high": 0.4,
            },
        ]
    )
    result = confidence_filtered_rank_reversals(
        pairwise,
        group_columns=["step", "subset"],
        entity_columns=("arm_a", "arm_b"),
    )
    assert result.loc[0, "n_informative_pairs"] == 0
    assert pd.isna(result.loc[0, "agreement_fraction"])


def test_specialist_comparison_plot_writes_svg_and_png(tmp_path):
    rows = []
    for subset in NON_DISTAL_SUBSETS:
        specialist = SPECIALIST_ARM[subset]
        for step in (500, 1000):
            for metric, value in (
                (AUPRC, 0.2 + step / 10_000),
                (CALIBRATED_BRIER, 0.1 - step / 100_000),
            ):
                rows.append(
                    {
                        "arm": specialist,
                        "step": step,
                        "subset": subset,
                        "metric": metric,
                        "value": value,
                        "ci_low": value - 0.01,
                        "ci_high": value + 0.01,
                    }
                )

    outputs = plot_exp232_specialist_auprc_vs_brier(
        pd.DataFrame(rows),
        tmp_path,
    )

    assert set(outputs) == {
        "plot_specialist_auprc_vs_brier_svg",
        "plot_specialist_auprc_vs_brier_png",
    }
    assert outputs["plot_specialist_auprc_vs_brier_svg"].stat().st_size > 0
    assert outputs["plot_specialist_auprc_vs_brier_png"].stat().st_size > 0
