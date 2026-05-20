"""Smoke tests for EnhancerSegmenter (random encoder, no pretrained weights)."""

import torch

from marin_dna.pipelines.enhancer_classification.model import ENCODER_OUTPUT_DIM
from marin_dna.pipelines.enhancer_segmentation.model import EnhancerSegmenter

BIN_SIZE = 128
NUM_BINS = 128
SEQ_LEN = BIN_SIZE * NUM_BINS  # 16384
BATCH = 2


def _random_batch() -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, 4, (BATCH, SEQ_LEN))
    x = torch.nn.functional.one_hot(indices, num_classes=4).float()
    y = torch.zeros(BATCH, NUM_BINS)
    y[0, 10:12] = 1.0
    return x, y


def test_forward_output_shape():
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False)
    model.eval()
    x, _ = _random_batch()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH, NUM_BINS)


def test_loss_computation():
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False, pos_weight=10.0)
    x, y = _random_batch()
    logits = model(x)
    loss = model.loss_fn(logits, y)
    assert loss.shape == ()
    assert loss.item() > 0


def test_freeze_backbone():
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=True)
    for param in model.encoder.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_head_includes_groupnorm():
    """The head applies channel-wise normalization before the final projection
    (issue #115 follow-up: stabilize grad norm)."""
    model = EnhancerSegmenter(weights_path=None, freeze_backbone=False)
    # Head is Sequential[GroupNorm, Conv1d].
    assert isinstance(model.head[0], torch.nn.GroupNorm)
    assert model.head[0].num_channels == 1536
    assert isinstance(model.head[-1], torch.nn.Conv1d)


def test_per_species_metric_per_genome():
    """Per-species AUPRC metrics are created one per genome."""
    model = EnhancerSegmenter(
        weights_path=None,
        freeze_backbone=False,
        genomes=["homo_sapiens", "mus_musculus"],
    )
    assert set(model.val_auprc_per_species.keys()) == {"homo_sapiens", "mus_musculus"}


def test_transformer_path_forward_and_shape():
    """n_transformer_layers>0 inserts a truncated tower and produces the
    same output shape as the encoder-only path."""
    model = EnhancerSegmenter(
        weights_path=None, freeze_backbone=False, n_transformer_layers=1
    )
    model.eval()
    x, _ = _random_batch()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH, NUM_BINS)
    # Tower exists and has exactly one block.
    assert model.tower is not None
    assert len(model.tower.tower.blocks) == 1
    # Species embedding is a single learnable 1536-vector (not nn.Embedding).
    assert model.tower.species_embed.shape == (ENCODER_OUTPUT_DIM,)
    assert model.tower.species_embed.requires_grad


def test_transformer_path_freeze_backbone():
    """freeze_backbone freezes encoder and tower (including species_embed);
    only the head remains trainable."""
    model = EnhancerSegmenter(
        weights_path=None, freeze_backbone=True, n_transformer_layers=1
    )
    for p in model.encoder.parameters():
        assert not p.requires_grad
    for p in model.tower.parameters():
        assert not p.requires_grad
    for p in model.head.parameters():
        assert p.requires_grad
