"""Tests for the AlphaGenome/Borzoi-style per-base head + loss (#259, alphagenome.py).

Pins the parts that must be exactly right: the target<->prediction scaling
roundtrip and soft-clip continuity, the Poisson-Multinomial loss, the per-base
forward contract (so the QTL scorer works unchanged), and a train step.
"""

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from marin_dna.pipelines.chrombpnet_eval.alphagenome import (
    AlphaGenomeLit,
    build_alphagenome_perbase,
    inverse_soft_clip,
    poisson_multinomial_loss,
    predictions_scaling,
    soft_clip_targets,
    targets_scaling,
)


def test_soft_clip_continuity_and_threshold():
    # Below the threshold (10) values pass through; at/above they are compressed
    # and the map is continuous at 10.
    t = torch.tensor([0.0, 5.0, 10.0, 40.0, 1000.0])
    c = soft_clip_targets(t)
    assert torch.allclose(c[:3], t[:3])  # <=10 untouched
    assert c[3] < t[3] and c[4] < t[4]  # >10 dampened
    # continuity at the threshold: soft_clip(10) == 10
    assert soft_clip_targets(torch.tensor([10.0])).item() == pytest.approx(10.0)
    # known value: t=1000 -> 2*sqrt(10*1000)-10 = 2*100-10 = 190
    assert c[4].item() == pytest.approx(190.0)


def test_soft_clip_inverse_roundtrip():
    t = torch.tensor([0.0, 3.0, 10.0, 53.2, 190.0, 5000.0])
    assert torch.allclose(inverse_soft_clip(soft_clip_targets(t)), t, atol=1e-4)


def test_scaling_roundtrip_recovers_targets():
    # predictions_scaling is the exact inverse of targets_scaling (DNase: no squash).
    track_mean = 7.3
    t = torch.rand(4, 64) * 500  # raw coverage, spans below/above 10*mean
    scaled = targets_scaling(t, track_mean, apply_squashing=False)
    back = predictions_scaling(scaled, track_mean, apply_squashing=False)
    assert torch.allclose(back, t, atol=1e-3)


def test_scaling_roundtrip_with_squashing():
    track_mean = 12.0
    t = torch.rand(4, 32) * 800
    scaled = targets_scaling(t, track_mean, apply_squashing=True)
    back = predictions_scaling(scaled, track_mean, apply_squashing=True)
    assert torch.allclose(back, t, atol=1e-2)


def test_poisson_multinomial_loss_finite_and_segments():
    pred = torch.rand(3, 256).abs() + 1e-3
    target = torch.randint(0, 20, (3, 256)).float()
    loss = poisson_multinomial_loss(pred, target, n_segments=8)
    assert torch.isfinite(loss) and loss.ndim == 0
    with pytest.raises(AssertionError):  # out_window not divisible by n_segments
        poisson_multinomial_loss(pred, target, n_segments=7)


@pytest.mark.parametrize("in_window,out_window", [(256, 256), (512, 256)])
def test_forward_contract_nonneg(in_window: int, out_window: int):
    # forward -> (pred[B,out_window]>=0, log_total[B,1]); reuses the QTL (profile,
    # log_counts) contract so score_log2fc reads log_total unchanged.
    model = build_alphagenome_perbase(out_window=out_window, n_filters=8, n_layers=3)
    pred, log_total = model(torch.randn(2, 4, in_window))
    assert pred.shape == (2, out_window) and log_total.shape == (2, 1)
    assert (pred >= 0).all() and torch.isfinite(log_total).all()


class _ToyDS(Dataset):
    def __init__(self, n: int = 12, in_window: int = 256, out_window: int = 256):
        self.n, self.in_window, self.out_window = n, in_window, out_window

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, self.in_window)
        oh[torch.randint(0, 4, (self.in_window,)), torch.arange(self.in_window)] = 1.0
        return {
            "onehot_seq": oh,
            "profile": torch.randint(0, 8, (self.out_window,)).float(),
        }


def test_trains_under_lit():
    model = build_alphagenome_perbase(out_window=256, n_filters=8, n_layers=3)
    lit = AlphaGenomeLit(model, track_mean=4.0, lr=1e-3, lr_scheduler="wsd")
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
