"""Contracts for the additive terminal random-validation VEP comparison."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from exp473_center_seeded_projection.random_validation_vep import (
    CHR18_POLICY,
    MATURE_MIRNA_SUBSET,
    RANDOM_POLICY,
    compare_matched_subset,
    compare_sge,
    exclude_mature_mirna_groups,
    validate_score_pair,
)
from exp473_center_seeded_projection.random_validation_vep_config import (
    BASELINE_EXPERIMENT_COMMIT,
    BASELINE_MODEL,
    BASELINE_RESULTS_ROOT,
    DATASET_NAMES,
    RANDOM_CHECKPOINT_ROOT,
    RELEVANT_SUBSETS,
    TERMINAL_STEP,
    build_random_validation_vep_config,
    model_name,
)


def _matched_scores(*, random_arm: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    position = 100
    for group in range(20):
        if random_arm:
            desired = [1.0 + group * 0.01, 0.1 * (group % 2), 0.0]
        else:
            desired = [0.05 * (group % 4), 0.2 + group * 0.01, 0.0]
        for label, score in zip((1, 0, 0), desired, strict=True):
            rows.append(
                {
                    "chrom": str(1 + 2 * (group % 5)),
                    "pos": position,
                    "ref": "A",
                    "alt": "C",
                    "label": label,
                    "subset": "missense_variant",
                    "match_group": f"group-{group}",
                    "llr_fwd": -score,
                    "llr_rc": -score,
                }
            )
            position += 1
    for label in (1, 0):
        rows.append(
            {
                "chrom": "1",
                "pos": position,
                "ref": "G",
                "alt": "T",
                "label": label,
                "subset": MATURE_MIRNA_SUBSET,
                "match_group": "mature-group",
                "llr_fwd": -0.5,
                "llr_rc": -0.5,
            }
        )
        position += 1
    return pd.DataFrame(rows)


def _sge_metrics(value_offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "AUPRC",
                "subset": subset,
                "accession": "_macro_avg_",
                "gene": "_macro_avg_",
                "score_type": "minus_llr_avg",
                "value": 0.2 + value_offset + index * 0.01,
                "se": 0.01,
                "n": 100 + index,
                "n_pos": 40 + index,
                "model": "model",
                "dataset": "sge",
                "split": "train",
            }
            for index, subset in enumerate(RELEVANT_SUBSETS["sge"])
        ]
    )


def test_random_validation_vep_config_is_terminal_only_and_additive() -> None:
    commit = "a" * 40
    config = build_random_validation_vep_config(commit)
    control = config["issue_473_random_validation_vep"]
    assert config["split"] == "train"
    assert control["held_out_access"] is False
    assert control["dataset_file"] == "train.parquet"
    assert control["results_root"] == (
        f"results/issue473/{commit}/random_validation_vep"
    )
    assert control["baseline_experiment_commit"] == BASELINE_EXPERIMENT_COMMIT
    assert control["baseline_model"] == BASELINE_MODEL
    assert control["baseline_results_root"] == BASELINE_RESULTS_ROOT
    assert control["relevant_subsets"] == {
        name: list(subsets) for name, subsets in RELEVANT_SUBSETS.items()
    }
    assert len(config["models"]) == 1
    model = config["models"][0]
    assert model["name"] == model_name(commit)
    assert model["datasets"] == list(DATASET_NAMES)
    assert model["gcs_path"] == (f"{RANDOM_CHECKPOINT_ROOT}/hf/step-{TERMINAL_STEP}")
    assert config["nuc_dep"]["models"] == []
    assert config["umap_embeddings"]["models"] == []
    assert config["ll_gap"]["models"] == []
    assert config["probe"]["models"] == []


def test_mature_mirna_exclusion_removes_the_complete_group() -> None:
    frame = _matched_scores(random_arm=False)
    filtered = exclude_mature_mirna_groups(frame)
    assert "mature-group" not in set(filtered["match_group"])
    assert MATURE_MIRNA_SUBSET not in set(filtered["subset"])
    assert len(filtered) == len(frame) - 2


def test_paired_mendelian_comparison_uses_identical_rows_and_shared_draws() -> None:
    chr18 = _matched_scores(random_arm=False)
    random = _matched_scores(random_arm=True)
    comparison, samples = compare_matched_subset(
        chr18,
        random,
        benchmark="mendelian_traits",
        subset="missense_variant",
        n_bootstrap=100,
        seed=473,
    )
    assert comparison["metric"].tolist() == ["auprc", "group_smd"]
    assert set(samples["policy"]) == {CHR18_POLICY, RANDOM_POLICY}
    assert set(samples["draw"]) == set(range(100))
    assert set(comparison["split"]) == {"train"}
    assert (
        comparison.loc[
            comparison["metric"] == "auprc", "delta_random_minus_chr18"
        ].item()
        > 0
    )
    assert np.isfinite(comparison["delta_ci_low"]).all()
    assert np.isfinite(comparison["delta_ci_high"]).all()


def test_paired_comparison_rejects_row_misalignment() -> None:
    chr18 = exclude_mature_mirna_groups(_matched_scores(random_arm=False))
    random = exclude_mature_mirna_groups(_matched_scores(random_arm=True))
    random.loc[0, "pos"] += 1
    with pytest.raises(AssertionError, match="not row-identical"):
        validate_score_pair(chr18, random)


def test_sge_comparison_selects_only_relevant_assay_macro_rows() -> None:
    comparison = compare_sge(_sge_metrics(0.0), _sge_metrics(0.03))
    assert comparison["subset"].tolist() == list(RELEVANT_SUBSETS["sge"])
    assert comparison["metric"].tolist() == ["auprc", "auprc"]
    assert comparison["delta_random_minus_chr18"].to_numpy() == pytest.approx(0.03)
    assert comparison["delta_ci_low"].isna().all()
    assert set(comparison["split"]) == {"train"}


def test_random_validation_workflow_is_an_isolated_three_cell_graph() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / "workflow" / "RandomValidationVep.smk").read_text()
    launcher = (root / "sky" / "random_validation_vep.yaml").read_text()
    assert "include:" not in workflow
    assert "workflow/Snakefile" not in workflow
    assert "len(DATASETS) == 3" in workflow
    assert "len(MODELS) == 1" in workflow
    assert "issue_473_random_validation_vep_score" in workflow
    assert "issue_473_random_validation_vep_analyze" in workflow
    assert "RandomValidationVep.smk" in launcher
    assert "issue_473_random_validation_vep_all" in launcher
    assert "Evaluation.smk" not in launcher
