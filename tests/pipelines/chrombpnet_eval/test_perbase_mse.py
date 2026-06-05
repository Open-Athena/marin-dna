"""Tests for the simplest per-base loss: MSE on log-counts per position (#259)."""

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from marin_dna.pipelines.chrombpnet_eval.perbase_mse import (
    PerBaseMSELogLit,
    build_perbase_mse,
)


@pytest.mark.parametrize("in_window,out_window", [(256, 256), (512, 256)])
def test_forward_contract(in_window: int, out_window: int):
    # forward -> (y[B,out_window], log_total[B,1]); reuses the QTL (profile,
    # log_counts) contract so score_log2fc reads log_total unchanged.
    model = build_perbase_mse(out_window=out_window, n_filters=8, n_layers=3)
    y, log_total = model(torch.randn(2, 4, in_window))
    assert y.shape == (2, out_window) and log_total.shape == (2, 1)
    assert torch.isfinite(y).all() and torch.isfinite(log_total).all()


def test_log_total_is_log_sum_expm1():
    # The QTL readout must be log(sum(expm1(y))) — per-base counts summed.
    torch.manual_seed(0)
    model = build_perbase_mse(out_window=256, n_filters=8, n_layers=3)
    y, log_total = model(torch.randn(2, 4, 256))
    expected = torch.log(
        torch.expm1(y.float()).clamp_min(0).sum(-1, keepdim=True) + 1e-7
    )
    assert torch.allclose(log_total, expected, atol=1e-5)


class _ToyDS(Dataset):
    def __init__(self, n: int = 12):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, 256)
        oh[torch.randint(0, 4, (256,)), torch.arange(256)] = 1.0
        return {"onehot_seq": oh, "profile": torch.randint(0, 8, (256,)).float()}


def test_trains_under_lit():
    model = build_perbase_mse(out_window=256, n_filters=8, n_layers=3)
    lit = PerBaseMSELogLit(model, lr=1e-3, lr_scheduler="wsd")
    trainer = L.Trainer(
        max_steps=3,
        limit_train_batches=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, DataLoader(_ToyDS(12), batch_size=4))
    m = {**trainer.callback_metrics, **trainer.logged_metrics}
    assert torch.isfinite(torch.as_tensor(float(m["train_loss_step"]))), m
