from __future__ import annotations

import json
from pathlib import Path

import train_seed289 as training


def test_fresh_seed_manifest_preserves_registered_single_arm() -> None:
    manifest = training.dry_run_manifest("dna-exp431-fresh-seed289", compile_llm=False)
    config = manifest["fixed_config"]

    assert manifest["issue"] == 431
    assert config["seed"] == 289
    assert config["reported_blocks"] == [10]
    assert config["budgets"] == [5_000_550]
    assert config["optimizer_steps"] == [1_961]
    assert config["d_sae"] == 15_360
    assert config["k"] == 64
    assert config["normalize_activations"] == "expected_average_only_in"
    assert config["lm_dtype"] == "bfloat16"
    assert config["sae_dtype"] == "float32"
    assert config["compile_llm"] is False
    assert len(config["checkpoint_thresholds"]) == 1
    assert config["first_checkpoint_batch_boundaries"][0] < training.SHORT_BUDGET
    assert training.arm_label(9, training.SHORT_BUDGET) == "block10-5m"


def test_reads_one_prefold_scaling_factor(tmp_path: Path) -> None:
    recovery = tmp_path / "recovery" / "block10"
    final = tmp_path / "final_5000550" / "block10"
    recovery.mkdir(parents=True)
    final.mkdir(parents=True)
    (recovery / "activation_scaler.json").write_text(
        json.dumps({"scaling_factor": 0.35})
    )
    (final / "activation_scaler.json").write_text(json.dumps({"scaling_factor": None}))

    assert training.recorded_activation_norm_scaling_factor(tmp_path) == 0.35


def test_runner_config_is_single_block_and_budget(tmp_path: Path) -> None:
    config = training.make_runner_config(
        checkpoint_path=tmp_path,
        run_name="dna-exp431-fresh-seed289",
        bos_token_id=2,
        log_to_wandb=False,
        compile_llm=False,
    )

    assert config.training_tokens == training.SHORT_BUDGET
    assert config.n_checkpoints == 1
    assert set(config.saes) == {"block10"}
    assert config.hook_names_per_sae == {"block10": "model.layers.9"}
