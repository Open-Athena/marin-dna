"""Tests for the zero-padded ('same') ChromBPNet variant (#259, samepad.py).

The point of this variant is that it decouples the output window and the param
count from the input window: 'same' padding preserves width, so out_window is a
free center-crop and a fixed n_layers gives a fixed param count at any context.
These tests pin exactly those properties, plus that it satisfies the ChromBPNet
forward contract and trains under ChromBPNetLit.
"""

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit
from marin_dna.pipelines.chrombpnet_eval.onehot import count_trainable_params
from marin_dna.pipelines.chrombpnet_eval.samepad import build_samepad_chrombpnet


@pytest.mark.parametrize("in_window,out_window", [(256, 256), (256, 128), (512, 256)])
def test_forward_contract_and_crop(in_window: int, out_window: int):
    # forward(onehot[B,4,L]) -> (profile[B,out_window], log_counts[B,1]), finite.
    model = build_samepad_chrombpnet(out_window=out_window, n_filters=8, n_layers=3)
    x = torch.randn(2, 4, in_window)
    profile, count = model(x)
    assert profile.shape == (2, out_window), profile.shape
    assert count.shape == (2, 1), count.shape
    assert torch.isfinite(profile).all() and torch.isfinite(count).all()


def test_params_independent_of_input_window():
    # The whole motivation: same n_layers -> same param count regardless of the
    # input window (convs are width-agnostic), and one model runs at any width.
    model = build_samepad_chrombpnet(out_window=128, n_filters=8, n_layers=3)
    n = count_trainable_params(model)
    for in_window in (256, 512, 1024):
        profile, count = model(torch.randn(2, 4, in_window))
        assert profile.shape == (2, 128) and count.shape == (2, 1)
    # a deeper model has strictly more params (n_layers is the size knob)
    deeper = build_samepad_chrombpnet(out_window=128, n_filters=8, n_layers=5)
    assert count_trainable_params(deeper) > n


def test_out_window_exceeding_input_raises():
    model = build_samepad_chrombpnet(out_window=512, n_filters=8, n_layers=2)
    with pytest.raises(AssertionError):
        model(torch.randn(2, 4, 256))  # out_window 512 > input 256


class _ToyDS(Dataset):
    def __init__(self, n: int = 8, in_window: int = 256, out_window: int = 256):
        self.n, self.in_window, self.out_window = n, in_window, out_window

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, self.in_window)
        oh[torch.randint(0, 4, (self.in_window,)), torch.arange(self.in_window)] = 1.0
        return {
            "onehot_seq": oh,
            "profile": torch.randint(0, 5, (self.out_window,)).float(),
        }


def test_trains_under_lit():
    # Drops into ChromBPNetLit unchanged and takes an optimizer step (finite loss).
    model = build_samepad_chrombpnet(out_window=256, n_filters=8, n_layers=3)
    lit = ChromBPNetLit(model, alpha=0.8, beta=1.0, lr=1e-3, lr_scheduler="wsd")
    trainer = L.Trainer(
        max_steps=3,
        limit_train_batches=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, DataLoader(_ToyDS(12), batch_size=4))
    metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
    assert torch.isfinite(torch.as_tensor(float(metrics["train_loss_step"]))), metrics
