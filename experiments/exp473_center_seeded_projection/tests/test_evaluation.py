"""Issue #473 development-only evaluation contracts."""

import numpy as np
import pandas as pd
import pytest
from exp473_center_seeded_projection.analyze_evals import (
    MENDELIAN_SUBSETS,
    seed_trigger_table,
    validate_policy_pair,
)
from exp473_center_seeded_projection.eval_config import (
    ARM_DATASETS,
    CHECKPOINT_STEPS,
    build_eval_config,
    model_name,
)
from exp473_center_seeded_projection.paired_metrics import (
    AUPRC,
    GROUP_SMD,
    group_smd,
    paired_policy_bootstrap,
    validate_matched_frame,
)


def _roots() -> dict[str, str]:
    return {arm: f"gs://test/immutable/{arm}-abc123" for arm in ARM_DATASETS}


def _matched() -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    labels: list[int] = []
    full: list[float] = []
    center: list[float] = []
    groups: list[int] = []
    for group in range(12):
        labels.extend([1, 0, 0])
        full.extend([1.0 + group * 0.1, 0.2 + group * 0.03, -0.1])
        center.extend([1.2 + group * 0.1, 0.1 + group * 0.02, -0.2])
        groups.extend([group, group, group])
    return (
        pd.Series(labels),
        pd.DataFrame({"full_window": full, "center_1": center}),
        pd.Series(groups),
    )


def test_eval_config_is_development_only_and_complete():
    config = build_eval_config(_roots(), experiment_commit="a" * 40)
    assert config["split"] == "train"
    assert config["issue_473"]["held_out_access"] is False
    assert len(config["models"]) == 4 * len(CHECKPOINT_STEPS)
    assert {dataset["name"] for dataset in config["datasets"]} == {
        "mendelian_traits",
        "complex_traits",
        "sge",
    }
    assert config["nuc_dep"]["models"] == []
    assert config["umap_embeddings"]["models"] == []
    assert config["ll_gap"]["models"] == []
    assert config["probe"]["models"] == []
    model = next(
        entry
        for entry in config["models"]
        if entry["name"] == model_name("cds_center_1", 500)
    )
    assert model["gcs_path"].endswith("/cds_center_1-abc123/hf/step-500")
    assert model["datasets"] == ["mendelian_traits", "sge"]


def test_eval_config_rejects_non_gcs_and_partial_roots():
    roots = _roots()
    roots["cds_full_window"] = "s3://wrong/checkpoint"
    with pytest.raises(ValueError, match="gs://"):
        build_eval_config(roots, experiment_commit="b" * 40)
    with pytest.raises(ValueError, match="exactly"):
        build_eval_config(
            {"cds_full_window": "gs://test/checkpoint"}, experiment_commit="b" * 40
        )


def test_group_smd_formula_and_positive_affine_invariance():
    labels, scores, groups = _matched()
    frame = validate_matched_frame(labels, scores, groups)
    gaps = []
    for _, group in frame.groupby("match_group", sort=False):
        positive = group.loc[group["label"] == 1, "full_window"].iloc[0]
        negative = group.loc[group["label"] == 0, "full_window"].mean()
        gaps.append(positive - negative)
    expected = np.mean(gaps) / np.std(gaps, ddof=1)
    assert group_smd(frame, "full_window") == pytest.approx(expected)

    transformed = scores * 7.0 + 11.0
    transformed_frame = validate_matched_frame(labels, transformed, groups)
    assert group_smd(transformed_frame, "full_window") == pytest.approx(expected)


def test_paired_bootstrap_uses_identical_draws_and_orients_delta():
    labels, scores, groups = _matched()
    result = paired_policy_bootstrap(labels, scores, groups, n_bootstrap=30, seed=473)
    assert set(result.point["metric"]) == {AUPRC, GROUP_SMD}
    assert set(result.deltas["metric"]) == {AUPRC, GROUP_SMD}
    assert (result.deltas["n_bootstrap"] == 30).all()
    assert (result.deltas["bootstrap_unit"] == "match_group").all()
    delta = result.deltas.set_index("metric")["delta_center_minus_full"]
    assert delta[AUPRC] == pytest.approx(0)  # both synthetic rankings are perfect
    assert delta[GROUP_SMD] > 0

    identical = paired_policy_bootstrap(
        labels,
        pd.DataFrame(
            {
                "full_window": scores["full_window"],
                "center_1": scores["full_window"],
            }
        ),
        groups,
        n_bootstrap=20,
        seed=11,
    )
    assert identical.deltas["delta_center_minus_full"].to_numpy() == pytest.approx(0)
    assert identical.deltas[["ci_low", "ci_high"]].to_numpy() == pytest.approx(0)


def test_policy_pair_must_have_exact_row_identity():
    rows = len(MENDELIAN_SUBSETS)
    full = pd.DataFrame(
        {
            "chrom": ["1"] * rows,
            "pos": list(range(rows)),
            "ref": ["A"] * rows,
            "alt": ["C"] * rows,
            "label": [index % 2 for index in range(rows)],
            "subset": list(MENDELIAN_SUBSETS),
            "match_group": list(range(rows)),
        }
    )
    validate_policy_pair(full, full.copy())
    changed = full.copy()
    changed.loc[0, "pos"] = 99
    with pytest.raises(AssertionError, match="row-identical"):
        validate_policy_pair(full, changed)


def test_seed_trigger_is_record_only():
    deltas = pd.DataFrame(
        {
            "region": ["cds"] * 3,
            "subset": ["missense_variant"] * 3,
            "metric": [AUPRC] * 3,
            "step": [500, 1_000, 1_500],
            "delta_center_minus_full": [0.01, 0.02, 0.03],
            "ci_low": [-0.01, -0.01, 0.005],
            "ci_high": [0.03, 0.04, 0.05],
        }
    )
    trigger = seed_trigger_table(deltas).iloc[0]
    assert bool(trigger["two_consecutive_same_direction"])
    assert bool(trigger["endpoint_interval_excludes_zero"])
    assert bool(trigger["additional_seed_trigger"])
