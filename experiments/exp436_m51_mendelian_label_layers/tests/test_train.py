from __future__ import annotations

import torch

from train import (
    BLOCK_INDICES,
    BUDGETS,
    LONG_BUDGET,
    NORM_ESTIMATE_BATCHES,
    SHORT_BUDGET,
    TRAIN_BATCH_TOKENS,
    arm_label,
    cast_multi_hook_activations,
    checkpoint_thresholds,
    dry_run_manifest,
    first_step_after,
    make_runner_config,
    training_windows_per_stream,
)


def test_multi_hook_activations_are_cast_to_sae_dtype() -> None:
    source = {"hook": torch.ones((2, 3), dtype=torch.bfloat16)}
    cast = cast_multi_hook_activations(source, torch.float32)
    assert source["hook"].dtype == torch.bfloat16
    assert cast["hook"].dtype == torch.float32
    assert torch.equal(cast["hook"], source["hook"].float())


def test_exact_registered_budgets_and_checkpoint_boundary() -> None:
    assert SHORT_BUDGET == 5_000_550
    assert LONG_BUDGET == 25_000_200
    assert [training_windows_per_stream(value) for value in BUDGETS] == [1_961, 9_804]
    thresholds = checkpoint_thresholds(LONG_BUDGET, 4)
    assert first_step_after(thresholds[0], TRAIN_BATCH_TOKENS) == SHORT_BUDGET


def test_arm_labels_are_unambiguous() -> None:
    assert {
        arm_label(block_index, budget)
        for block_index in BLOCK_INDICES
        for budget in BUDGETS
    } == {
        "block01-5m",
        "block01-25m",
    }


def test_runner_config_constructs_registered_shared_forward(tmp_path) -> None:
    cfg = make_runner_config(
        checkpoint_path=tmp_path / "checkpoints",
        run_name="dna-exp436-test",
        bos_token_id=2,
        log_to_wandb=False,
        compile_llm=True,
    )
    assert cfg.training_tokens == LONG_BUDGET
    assert cfg.train_batch_size_tokens == TRAIN_BATCH_TOKENS
    assert cfg.n_batches_for_norm_estimate == NORM_ESTIMATE_BATCHES
    assert cfg.compile_llm and cfg.autocast_lm and not cfg.autocast
    assert cfg.activations_mixing_fraction == 0
    assert cfg.prefetch_llm_batches == 2
    assert len(cfg.saes) == len(BLOCK_INDICES)
    assert len(set(cfg.hook_names_per_sae.values())) == len(BLOCK_INDICES)
    assert all(
        sae.normalize_activations == "expected_average_only_in"
        for sae in cfg.saes.values()
    )


def test_dry_run_manifest_exposes_normalization_and_performance_contract() -> None:
    manifest = dry_run_manifest("dna-exp436-test", compile_llm=True)
    fixed = manifest["fixed_config"]
    assert fixed["budgets"] == list(BUDGETS)
    assert fixed["reported_blocks"] == [1]
    assert fixed["normalization_observations_per_layer"] == (
        NORM_ESTIMATE_BATCHES * TRAIN_BATCH_TOKENS
    )
    assert fixed["lm_dtype"] == "bfloat16"
    assert fixed["sae_dtype"] == "float32"
    assert fixed["compile_llm"] is True
    assert fixed["model_use_cache"] is False
    assert fixed["multi_hook_activation_dtype_adapter"] == "configured_store_dtype"
    assert manifest["data"]["orientations"] == ["forward", "reverse_complement"]
