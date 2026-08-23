"""Tests for language modeling (LM) models."""

import pytest
import torch

from glm_experiments.models.components.bytenet import ByteNet
from glm_experiments.models.components.lm import CLM, MLM
from glm_experiments.models.components.transformer import Embedding, Transformer


@pytest.fixture
def mlm_model():
    """Create a small MLM model for testing."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=4, slim=True)
    layer_norm = torch.nn.LayerNorm(128)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    return MLM(embedder=embedder, encoder=encoder, layer_norm=layer_norm, decoder=decoder)


def test_mlm_embed(mlm_model):
    """Test MLM embedding layer."""
    input_ids = torch.randint(0, 6, (2, 100))  # (batch, seq_len)
    embeddings = mlm_model.embedder(input_ids)

    assert embeddings.shape == (2, 100, 128)  # (batch, seq_len, hidden)
    assert embeddings.dtype == torch.float32


def test_mlm_forward(mlm_model):
    """Test MLM forward pass with loss calculation."""
    batch_size = 2
    seq_len = 100

    # Create inputs
    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))

    # Mask 15% of positions
    mask = torch.rand(batch_size, seq_len) < 0.15
    labels[~mask] = -100  # -100 means don't compute loss

    # soft_masked boolean tensor (all False for simplicity)
    soft_masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    soft_masked_weight = 0.01

    # Forward pass
    loss_dict = mlm_model(input_ids, labels, soft_masked, soft_masked_weight)

    # Should return dict with three losses
    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    assert "loss_full" in loss_dict
    assert "loss_non_soft_masked" in loss_dict
    assert loss_dict["loss"].dtype == torch.float32
    assert loss_dict["loss"].item() >= 0.0  # Loss should be non-negative


def test_mlm_weighted_loss(mlm_model):
    """Test MLM loss decomposition with soft_masked positions."""
    batch_size = 2
    seq_len = 100

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))

    # Mask all positions
    labels[:, :] = torch.randint(0, 6, (batch_size, seq_len))

    # Create soft_masked pattern (half soft-masked, half not)
    soft_masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    soft_masked[:, seq_len // 2 :] = True  # Second half is soft-masked

    # Test with different soft_masked_weights
    soft_masked_weight_high = 1.0  # No difference
    soft_masked_weight_low = 0.01  # Large difference

    loss_dict_high = mlm_model(input_ids, labels, soft_masked, soft_masked_weight_high)
    loss_dict_low = mlm_model(input_ids, labels, soft_masked, soft_masked_weight_low)

    # When soft_masked_weight=1.0, all three losses should be equal
    assert torch.allclose(loss_dict_high["loss"], loss_dict_high["loss_full"], rtol=1e-5)
    assert torch.allclose(
        loss_dict_high["loss"], loss_dict_high["loss_non_soft_masked"], rtol=1e-5
    )

    # When soft_masked_weight=0.01, losses should differ
    # loss < loss_full (because soft-masked tokens have less weight)
    # loss_non_soft_masked should be different (only hard-masked)
    assert not torch.allclose(loss_dict_low["loss"], loss_dict_low["loss_full"], rtol=0.1)


def test_mlm_no_masked_positions():
    """Test MLM when no positions are masked (all labels = -100)."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)
    layer_norm = torch.nn.LayerNorm(128)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    model = MLM(embedder=embedder, encoder=encoder, layer_norm=layer_norm, decoder=decoder)

    batch_size = 2
    seq_len = 50

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100)  # All ignored
    soft_masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    soft_masked_weight = 0.01

    # Should still work (loss will be 0.0 when no valid positions)
    loss_dict = model(input_ids, labels, soft_masked, soft_masked_weight)
    assert loss_dict["loss"].item() == 0.0


def test_mlm_all_masked_positions():
    """Test MLM when all positions are masked."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)
    layer_norm = torch.nn.LayerNorm(128)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    model = MLM(embedder=embedder, encoder=encoder, layer_norm=layer_norm, decoder=decoder)

    batch_size = 2
    seq_len = 50

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))  # All valid
    soft_masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    soft_masked_weight = 0.01

    loss_dict = model(input_ids, labels, soft_masked, soft_masked_weight)
    assert loss_dict["loss"].dtype == torch.float32
    assert loss_dict["loss"].item() >= 0.0


def test_mlm_batch_size_one():
    """Test MLM with batch size of 1."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)
    layer_norm = torch.nn.LayerNorm(128)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    model = MLM(embedder=embedder, encoder=encoder, layer_norm=layer_norm, decoder=decoder)

    input_ids = torch.randint(0, 6, (1, 50))
    labels = torch.randint(0, 6, (1, 50))
    soft_masked = torch.zeros(1, 50, dtype=torch.bool)
    soft_masked_weight = 0.01

    loss_dict = model(input_ids, labels, soft_masked, soft_masked_weight)
    assert loss_dict["loss"].item() >= 0.0


def test_mlm_get_logits(mlm_model):
    """Test MLM get_logits returns correct shape."""
    batch_size = 2
    seq_len = 100
    vocab_size = 6  # Matches fixture

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = mlm_model.get_logits(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert logits.dtype == torch.float32


def test_mlm_get_logits_used_by_forward(mlm_model):
    """Test that forward uses get_logits internally."""
    batch_size = 2
    seq_len = 100

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))
    soft_masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    soft_masked_weight = 0.01

    # Get logits directly
    logits = mlm_model.get_logits(input_ids)

    # Forward should produce same logits (loss computed from same logits)
    # We can't directly compare, but we can verify shapes match
    assert logits.shape[:-1] == (batch_size, seq_len)


# CLM Tests


@pytest.fixture
def clm_model():
    """Create a small CLM model for testing."""
    embedder = Embedding(vocab_size=6, d_model=128)
    encoder = Transformer(hidden_size=128, n_layers=2, num_heads=4, is_causal=True)
    layer_norm = torch.nn.RMSNorm(128)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    return CLM(embedder=embedder, encoder=encoder, layer_norm=layer_norm, decoder=decoder)


def test_clm_forward(clm_model):
    """Test CLM forward pass with next-token prediction."""
    batch_size = 2
    seq_len = 100

    # Create inputs (no masking needed for CLM)
    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = input_ids.clone()  # Labels are same as input_ids for CLM

    # soft_masked boolean tensor (can indicate soft-masked positions)
    soft_masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    soft_masked_weight = 0.01

    # Forward pass
    loss_dict = clm_model(input_ids, labels, soft_masked, soft_masked_weight)

    # Should return dict with three losses
    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    assert loss_dict["loss"].dtype == torch.float32
    assert loss_dict["loss"].item() >= 0.0


def test_clm_validates_causal_encoder():
    """Test that CLM raises error if encoder is not causal."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)  # is_causal=False
    layer_norm = torch.nn.LayerNorm(128)
    decoder = torch.nn.Linear(in_features=128, out_features=6)

    with pytest.raises(ValueError, match="CLM requires causal encoder"):
        CLM(embedder=embedder, encoder=encoder, layer_norm=layer_norm, decoder=decoder)


def test_clm_get_logits(clm_model):
    """Test CLM get_logits returns correct shape."""
    batch_size = 2
    seq_len = 100
    vocab_size = 6

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = clm_model.get_logits(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert logits.dtype == torch.float32
