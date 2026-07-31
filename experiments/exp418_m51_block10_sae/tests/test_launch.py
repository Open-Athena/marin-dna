from __future__ import annotations

from launch import (
    BUFFER_CONTEXT_BATCHES,
    CONTEXT_TOKENS,
    HUMAN_FASTA_URI,
    MODEL_REVISION,
    N_SOURCES,
    SAELENS_REVISION,
    SEED,
    TRAIN_BATCH_TOKENS,
    _dry_run_manifest,
    _runner_config,
    tier_config,
)


def test_training_batches_and_budgets_are_exact() -> None:
    assert TRAIN_BATCH_TOKENS == 1_275
    assert BUFFER_CONTEXT_BATCHES * CONTEXT_TOKENS >= TRAIN_BATCH_TOKENS
    for name, expected_steps in (("wiring", 785), ("micro", 3_922)):
        tier = tier_config(name)  # type: ignore[arg-type]
        assert tier.optimizer_steps == expected_steps
        assert tier.optimizer_steps == tier.budget.windows_per_source
        assert tier.budget.total_windows == expected_steps * N_SOURCES


def test_dry_run_manifest_pins_scientific_inputs() -> None:
    manifest = _dry_run_manifest(tier_config("wiring"), "test-run")
    fixed = manifest["fixed_config"]
    assert fixed["model_revision"] == MODEL_REVISION
    assert fixed["saelens_revision"] == SAELENS_REVISION
    assert fixed["seed"] == SEED
    assert fixed["activations_mixing_fraction"] == 0.0
    assert manifest["interpretation_boundary"]["post_training"].startswith(
        "coordinate-clean held-out human GRCh38"
    )
    assert manifest["interpretation_boundary"]["human_fasta_uri"] == HUMAN_FASTA_URI


def test_pinned_saelens_runner_config_constructs(tmp_path) -> None:
    tier = tier_config("wiring")
    cfg = _runner_config(
        tier=tier,
        output_path=tmp_path / "sae",
        checkpoint_path=tmp_path / "checkpoints",
        run_name="dna-exp418-test",
        bos_token_id=2,
        log_to_wandb=False,
        n_checkpoints=2,
    )
    assert cfg.hook_name == "model.layers.9"
    assert cfg.training_tokens == 1_000_875
    assert cfg.train_batch_size_tokens == 1_275
    assert cfg.exclude_special_tokens == [2]
    assert cfg.sae.architecture() == "batchtopk"
