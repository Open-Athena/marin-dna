"""Build the one-hot ChromBPNet baseline (issue #241).

A thin, testable constructor around the vendored ChromBPNet: an accessibility
``BPNet`` (dilated CNN over 4-channel one-hot, official 512-filter / 8-layer
config) plus a **frozen, pretrained** Tn5/DNase bias ``BPNet`` loaded from
ARSENAL's ``bias_model_scaled.h5``. This is the faithful one-hot ChromBPNet —
no gLM, no architectural change — used to validate the training pipeline and as
the supervised baseline (cf. the embedding arm, #243).
"""

from __future__ import annotations

import torch

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.bpnet import BPNet
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.chrombpnet import ChromBPNet
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_config import (
    ChromBPNetConfig,
)


def load_frozen_bias(model: ChromBPNet, bias_h5: str) -> None:
    """Replace ``model.bias`` with the pretrained bias from a Keras ``.h5`` and
    freeze it.

    ``ChromBPNet.__init__`` builds a *fresh, untrained* bias ``BPNet``; real
    training needs the GC-matched bias model trained on non-peak regions, held
    frozen so the accessibility tower learns motifs rather than enzyme bias.
    ``BPNet.from_keras`` reconstructs the bias architecture from the stored
    weights (h5py-only; no TensorFlow).
    """
    model.bias = BPNet.from_keras(bias_h5, name="bias")
    for p in model.bias.parameters():
        p.requires_grad = False


def build_onehot_chrombpnet(
    *,
    bias_h5: str | None = None,
    out_dim: int = 1000,
    n_filters: int = 512,
    n_layers: int = 8,
    conv1_kernel_size: int = 21,
    profile_kernel_size: int = 75,
) -> ChromBPNet:
    """Construct a one-hot ChromBPNet.

    Args:
        bias_h5: path to ARSENAL's ``bias_model_scaled.h5``. When given, the
            pretrained bias is loaded and frozen (see :func:`load_frozen_bias`).
            ``None`` keeps ChromBPNet's fresh untrained bias — only for code-path
            smoke tests on synthetic data, **never** a real training run.
        out_dim: profile output width (central window, 1000 bp).
        n_filters / n_layers / conv1_kernel_size / profile_kernel_size: the
            accessibility-tower architecture. Defaults are official one-hot
            ChromBPNet (512 filters, 8 dilated layers); shrink for fast tests.

    Returns:
        A ``ChromBPNet`` whose ``forward(onehot[B,4,2114])`` returns
        ``(profile_logits[B,out_dim], log_counts[B,1])``.
    """
    config = ChromBPNetConfig(
        out_dim=out_dim,
        n_filters=n_filters,
        n_layers=n_layers,
        conv1_kernel_size=conv1_kernel_size,
        profile_kernel_size=profile_kernel_size,
        n_outputs=1,
        n_control_tracks=0,
    )
    model = ChromBPNet(config)
    if bias_h5 is not None:
        load_frozen_bias(model, bias_h5)
    return model


def count_trainable_params(model: torch.nn.Module) -> int:
    """Number of trainable (``requires_grad``) parameters — used by the driver
    to print model size and to assert the bias is frozen."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
