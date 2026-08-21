from __future__ import annotations

import pandas as pd
import pytest
from exp473_center_seeded_projection.native_validation_replay import (
    EXPECTED_ROPE_SCALING,
    add_case_weighted_loss,
    aggregate_case_weighted_loss,
    validate_dual_schema_rope,
    validate_validation_frame,
)
from exp473_center_seeded_projection.native_validation_replay_config import (
    NATIVE_WANDB_LOSSES,
    build_native_validation_replay_config,
)


def _rope_config() -> dict:
    return {
        "transformers_version": "5.12.1",
        "rope_parameters": {"rope_theta": 500_000, **EXPECTED_ROPE_SCALING},
        "rope_theta": 500_000,
        "rope_scaling": dict(EXPECTED_ROPE_SCALING),
    }


def test_dual_schema_rope_requires_equivalent_transformers_4_and_5_fields() -> None:
    assert validate_dual_schema_rope(_rope_config()) == {
        "rope_theta": 500_000.0,
        "rope_scaling": EXPECTED_ROPE_SCALING,
    }
    missing_mirror = _rope_config()
    del missing_mirror["rope_scaling"]
    with pytest.raises(ValueError, match="Transformers-4"):
        validate_dual_schema_rope(missing_mirror)
    conflict = _rope_config()
    conflict["rope_theta"] = 10_000
    with pytest.raises(ValueError, match="conflicting RoPE theta"):
        validate_dual_schema_rope(conflict)


def test_validation_contract_is_exact_chr18_original_orientation() -> None:
    frame = pd.DataFrame(
        {
            "query_name": ["q1", "q2"],
            "species": ["Human", "Mouse"],
            "augmentation": ["+", "+"],
            "source_chrom": ["chr18", "chr18"],
            "source_start": [10, 20],
            "source_end": [265, 275],
            "region_label": ["cds", "cds"],
            "sequence": ["A" * 255, "a" * 255],
            "clade": ["primates", "mammals"],
            "family": ["Hominidae", "Muridae"],
            "taxonomy_id": [9606, 10090],
        }
    )
    validated = validate_validation_frame(
        frame, arm="cds_center_1", region="cds", expected_rows=2
    )
    assert validated["row_id"].tolist() == ["q1|Human|+", "q2|Mouse|+"]
    held_out_drift = frame.copy()
    held_out_drift.loc[0, "source_chrom"] = "chr17"
    with pytest.raises(AssertionError):
        validate_validation_frame(
            held_out_drift, arm="cds_center_1", region="cds", expected_rows=2
        )


def test_case_weighted_loss_matches_training_weighting() -> None:
    atoms = pd.DataFrame(
        {
            "ll_sum_upper": [-2.0, -3.0],
            "ll_sum_lower": [-4.0, -5.0],
            "n_upper": [2, 3],
            "n_lower": [4, 5],
        }
    )
    weighted = add_case_weighted_loss(atoms)
    expected = (2.0 + 0.01 * 4.0 + 3.0 + 0.01 * 5.0) / (2 + 0.01 * 4 + 3 + 0.01 * 5)
    assert aggregate_case_weighted_loss(weighted) == pytest.approx(expected)


def test_replay_config_is_only_three_new_arms_at_two_checkpoints() -> None:
    commit = "a" * 40
    config = build_native_validation_replay_config(diagnostic_commit=commit)
    assert config["interpretation_allowed"] is False
    assert config["vep_held_out_access"] is False
    assert len(config["models"]) == 6
    assert {
        (model["arm"], model["step"], model["native_wandb_loss"])
        for model in config["models"]
    } == {(arm, step, loss) for (arm, step), loss in NATIVE_WANDB_LOSSES.items()}
    assert "cds_full_window" not in {model["arm"] for model in config["models"]}
    assert all(
        source["repo"].startswith("marin-dna/") for source in config["sources"].values()
    )
