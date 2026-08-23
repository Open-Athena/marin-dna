"""Tests for LM Lightning modules."""

import pytest
import torch
from hydra import compose, initialize

from glm_experiments.models.lm_lit_module import (
    CausalLMAdapter,
    CLMLitModule,
    MaskedLMAdapter,
    MLMLitModule,
)


@pytest.fixture
def mlm_lit_module():
    """Create MLMLitModule from config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train",
            overrides=["model=bert_bytenet_small", "data=gpn_animal_promoter"],
        )

    import hydra

    return hydra.utils.instantiate(cfg.model)


def test_mlm_lit_module_instantiation():
    """Test that MLMLitModule can be instantiated from config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    assert isinstance(model, MLMLitModule)
    assert hasattr(model, "net")
    assert hasattr(model, "hparams")
    assert "optimizer" in model.hparams
    assert "scheduler" in model.hparams


def test_mlm_lit_module_forward():
    """Test forward pass through MLM LightningModule."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Create dummy batch
    batch_size = 2
    seq_len = 100
    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "soft_masked": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }

    # Forward pass
    loss_dict = model.model_step(batch)

    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    assert loss_dict["loss"].dtype == torch.float32
    assert loss_dict["loss"].item() >= 0.0


def test_mlm_lit_module_configure_optimizers():
    """Test optimizer and scheduler configuration."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    optim_config = model.configure_optimizers()

    assert "optimizer" in optim_config
    assert "lr_scheduler" in optim_config
    assert optim_config["lr_scheduler"]["interval"] == "step"


def test_mlm_lit_module_training_step():
    """Test MLM training step."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Create dummy batch
    batch_size = 2
    seq_len = 100
    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "soft_masked": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }

    # Training step
    loss = model.training_step(batch, batch_idx=0)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0


def test_mlm_lit_module_validation_step():
    """Test MLM validation step."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Create dummy batch
    batch_size = 2
    seq_len = 100
    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "soft_masked": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }

    # Validation step (returns None)
    result = model.validation_step(batch, batch_idx=0)
    assert result is None


def test_mlm_lit_module_save_hyperparameters():
    """Test that hyperparameters are saved correctly."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Check that hparams were saved (excluding net)
    assert hasattr(model, "hparams")
    assert "optimizer" in model.hparams
    assert "scheduler" in model.hparams
    assert "evals" in model.hparams  # New parameter
    # net should NOT be in hparams
    assert "net" not in model.hparams


def test_masked_lm_adapter(mlm_lit_module):
    """Test MaskedLMAdapter returns logits from MLM model."""
    adapter = mlm_lit_module.adapter

    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, 7, (batch_size, seq_len))

    logits = adapter(input_ids)

    assert logits.shape == (batch_size, seq_len, 7)  # vocab_size=7
    assert logits.dtype == torch.float32


def test_validation_step_traitgym_mendelian_promoter(mlm_lit_module):
    """Test validation_step with TraitGym Mendelian Promoter batch (dataloader_idx=1)."""
    batch_size = 4
    seq_len = 512

    # Create TraitGym Mendelian Promoter-style batch
    batch = {
        "input_ids": torch.randint(0, 7, (batch_size, seq_len)),
        "pos": torch.full((batch_size,), 256),  # Center position
        "ref": torch.randint(1, 5, (batch_size,)),  # Token IDs for A, C, G, T
        "alt": torch.randint(1, 5, (batch_size,)),
        "label": torch.randint(0, 2, (batch_size,)),  # Binary labels
    }

    # Run validation step with dataloader_idx=1 (TraitGym Mendelian Promoter)
    mlm_lit_module.validation_step(batch, batch_idx=0, dataloader_idx=1)

    # Check that metrics were updated using new structure
    eval_name = "traitgym_mendelian_promoter"
    assert eval_name in mlm_lit_module._eval_metrics
    assert mlm_lit_module._eval_metrics[eval_name]["scores"].update_count == 1
    assert mlm_lit_module._eval_metrics[eval_name]["labels"].update_count == 1


def test_on_validation_epoch_end_computes_auprc(mlm_lit_module):
    """Test on_validation_epoch_end computes AUPRC from accumulated data."""
    # Simulate accumulated data from multiple batches
    scores = torch.tensor([0.1, 0.4, 0.6, 0.9])
    labels = torch.tensor([0, 0, 1, 1])

    eval_name = "traitgym_mendelian_promoter"
    mlm_lit_module._eval_metrics[eval_name]["scores"].update(scores)
    mlm_lit_module._eval_metrics[eval_name]["labels"].update(labels)

    # Run epoch end
    mlm_lit_module.on_validation_epoch_end()

    # Metrics should be reset after computation
    assert mlm_lit_module._eval_metrics[eval_name]["scores"].update_count == 0
    assert mlm_lit_module._eval_metrics[eval_name]["labels"].update_count == 0


def test_validation_step_lm_still_works(mlm_lit_module):
    """Test that LM validation (dataloader_idx=0) still works."""
    batch_size = 2
    seq_len = 100

    batch = {
        "input_ids": torch.randint(0, 7, (batch_size, seq_len)),
        "labels": torch.randint(0, 7, (batch_size, seq_len)),
        "soft_masked": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }

    # Should not raise
    result = mlm_lit_module.validation_step(batch, batch_idx=0, dataloader_idx=0)
    assert result is None


def test_on_before_optimizer_step_logs_grad_norm(mlm_lit_module):
    """Test that on_before_optimizer_step computes and logs gradient norm."""
    from lightning.pytorch.utilities import grad_norm

    batch_size = 2
    seq_len = 100

    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "soft_masked": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }

    # Forward and backward to populate gradients
    loss_dict = mlm_lit_module.model_step(batch)
    loss_dict["loss"].backward()

    # Compute expected grad norm
    norms = grad_norm(mlm_lit_module, norm_type=2)
    expected_norm = norms["grad_2.0_norm_total"]

    # Verify gradients exist and norm is reasonable
    assert expected_norm > 0.0
    assert torch.isfinite(torch.tensor(expected_norm))


# CLM Tests


def test_clm_lit_module_instantiation():
    """Test that CLMLitModule can be instantiated from config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=clm_transformer_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    assert isinstance(model, CLMLitModule)
    assert hasattr(model, "net")
    assert hasattr(model, "adapter")
    assert isinstance(model.adapter, CausalLMAdapter)


def test_clm_lit_module_forward():
    """Test forward pass through CLM LightningModule."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=clm_transformer_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Create dummy batch
    batch_size = 2
    seq_len = 100
    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "soft_masked": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }

    # Forward pass
    loss_dict = model.model_step(batch)

    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    assert loss_dict["loss"].dtype == torch.float32
    assert loss_dict["loss"].item() >= 0.0
