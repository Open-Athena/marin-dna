"""Issue #473 development-only evaluation contracts."""

import json
from pathlib import Path

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
from exp473_center_seeded_projection.intersection_loss import (
    SCORED_COLUMNS,
    SPLIT,
    analyze_loss_scores,
    case_weighted_atoms,
    paired_loss_bootstrap,
    validate_intersection_frame,
)
from exp473_center_seeded_projection.intersection_loss_config import (
    PRODUCER_COMMIT,
    PRODUCER_CONFIG_SHA256,
    build_intersection_loss_config,
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


def _intersection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["row-a", "row-b", "row-c", "row-d"],
            "query_name": ["anchor-a", "anchor-a", "anchor-b", "anchor-b"],
            "species": ["species-1", "species-2", "species-1", "species-2"],
            "source_chrom": ["chr18"] * 4,
            "source_start": [0, 0, 10, 10],
            "source_end": [255, 255, 265, 265],
            "region_label": ["cds"] * 4,
            "sequence": ["A" * 255, "a" * 255, "AC" * 127 + "A", "T" * 255],
        }
    )


def _intersection_scores(
    policy: str, loss: float, *, region: str = "cds", step: int = 500
) -> pd.DataFrame:
    source = _intersection_frame()
    arm = f"{region}_{policy}"
    frame = source[["row_id", "query_name", "species", "source_chrom"]].copy()
    frame["region"] = region
    frame["policy"] = policy
    frame["arm"] = arm
    frame["step"] = step
    frame["split"] = SPLIT
    frame["nll_numerator"] = loss * 10.0
    frame["effective_tokens"] = 10.0
    frame["case_weighted_nll"] = loss
    assert tuple(frame.columns) == SCORED_COLUMNS
    return frame


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


def test_intersection_loss_config_is_pinned_complete_and_unlabeled():
    config = build_intersection_loss_config(_roots(), experiment_commit="c" * 40)
    assert config["producer_commit"] == PRODUCER_COMMIT
    assert config["producer_config_sha256"] == PRODUCER_CONFIG_SHA256
    assert config["split"] == SPLIT
    assert config["vep_held_out_access"] is False
    assert len(config["models"]) == 4 * len(CHECKPOINT_STEPS)
    assert len(config["sources"]) == 4
    assert {
        (source["region"], source["policy"]) for source in config["sources"].values()
    } == {
        (region, policy)
        for region in ("cds", "enhancer")
        for policy in ("full_window", "center_1")
    }
    assert all(
        source["uri"].startswith(
            "s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/v1/"
            f"{PRODUCER_COMMIT}/{PRODUCER_CONFIG_SHA256}/full/"
        )
        for source in config["sources"].values()
    )


def test_intersection_frame_contract_and_case_weighted_loss():
    frame = _intersection_frame()
    validated = validate_intersection_frame(frame, region="cds")
    assert validated["row_id"].tolist() == sorted(frame["row_id"])

    wrong_chrom = frame.copy()
    wrong_chrom.loc[0, "source_chrom"] = "chr17"
    with pytest.raises(AssertionError):
        validate_intersection_frame(wrong_chrom, region="cds")
    wrong_length = frame.copy()
    wrong_length.loc[0, "sequence"] = "A" * 254
    with pytest.raises(AssertionError, match="255 bp"):
        validate_intersection_frame(wrong_length, region="cds")

    weighted = case_weighted_atoms(
        pd.DataFrame(
            {
                "ll_sum_upper": [-4.0],
                "ll_sum_lower": [-10.0],
                "n_upper": [2],
                "n_lower": [10],
            }
        )
    )
    assert weighted.loc[0, "nll_numerator"] == pytest.approx(4.1)
    assert weighted.loc[0, "effective_tokens"] == pytest.approx(2.1)
    assert weighted.loc[0, "case_weighted_nll"] == pytest.approx(4.1 / 2.1)


def test_intersection_bootstrap_is_paired_and_negative_favors_center():
    full = _intersection_scores("full_window", 1.0)
    center = _intersection_scores("center_1", 0.8)
    points, samples, delta = paired_loss_bootstrap(
        full, center, n_bootstrap=40, seed=473
    )
    assert set(points["policy"]) == {"full_window", "center_1"}
    assert len(samples) == 80
    assert delta.loc[0, "delta_center_minus_full"] == pytest.approx(-0.2)
    assert delta.loc[0, "ci_low"] == pytest.approx(-0.2)
    assert delta.loc[0, "ci_high"] == pytest.approx(-0.2)
    assert delta.loc[0, "probability_center_better"] == 1.0
    assert delta.loc[0, "direction"] == "negative_favors_center_1"

    mismatched = center.copy()
    mismatched.loc[0, "row_id"] = "different"
    with pytest.raises(AssertionError, match="exactly paired"):
        paired_loss_bootstrap(full, mismatched, n_bootstrap=20, seed=473)


def test_intersection_workflow_is_additive_and_rule_isolated():
    root = Path(__file__).parents[1]
    snakefile = (root / "workflow" / "IntersectionLoss.smk").read_text()
    launcher = (root / "sky" / "intersection_loss.yaml").read_text()
    assert "vep_held_out_access" in snakefile
    assert "issue_473_intersection_loss" in launcher
    assert "--allowed-rules" in launcher
    assert "compute_scores" not in launcher
    assert "include:" not in snakefile


def test_intersection_analysis_requires_and_writes_complete_matrix(tmp_path: Path):
    score_paths: list[str] = []
    scores = tmp_path / "scores"
    scores.mkdir()
    for region in ("cds", "enhancer"):
        for policy, loss in (("full_window", 1.0), ("center_1", 0.8)):
            for step in CHECKPOINT_STEPS:
                path = scores / f"{region}-{policy}-{step}.parquet"
                _intersection_scores(policy, loss, region=region, step=step).to_parquet(
                    path, index=False
                )
                score_paths.append(str(path))

    output = tmp_path / "analysis"
    analyze_loss_scores(score_paths, output, n_bootstrap=10, seed=473)
    points = pd.read_parquet(output / "paired_loss_metrics.parquet")
    samples = pd.read_parquet(output / "paired_loss_bootstrap_samples.parquet")
    deltas = pd.read_parquet(output / "paired_loss_deltas.parquet")
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(points) == 40
    assert len(samples) == 400
    assert len(deltas) == 20
    assert set(deltas["region"]) == {"cds", "enhancer"}
    assert deltas["delta_center_minus_full"].to_numpy() == pytest.approx(-0.2)
    assert manifest["vep_held_out_access"] is False
    assert manifest["source_kind"] == (
        "unlabeled chromosome-18 projection intersection"
    )
    assert (
        "Negative `center_1 - full_window` deltas favor `center_1`."
        in (output / "summary.md").read_text()
    )
