"""Tests for the EMA-of-weights knob (#259, ema.py)."""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from marin_dna.pipelines.chrombpnet_eval.ema import EMA, EMACallback
from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit
from marin_dna.pipelines.chrombpnet_eval.samepad import build_samepad_chrombpnet


def test_ema_update_and_average_parameters():
    model = build_samepad_chrombpnet(out_window=256, n_filters=8, n_layers=2)
    ema = EMA(model, decay=0.5)
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():  # perturb live weights to p0 + 1
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)  # shadow = 0.5*p0 + 0.5*(p0+1) = p0 + 0.5
    with ema.average_parameters(model):  # swaps shadow (p0+0.5) into the model
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert torch.allclose(p, p0[n] + 0.5, atol=1e-5), n
    for n, p in model.named_parameters():  # restored to the live weights (p0+1)
        if p.requires_grad:
            assert torch.allclose(p, p0[n] + 1.0, atol=1e-5), n


class _ToyDS(Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, 256)
        oh[torch.randint(0, 4, (256,)), torch.arange(256)] = 1.0
        return {"onehot_seq": oh, "profile": torch.randint(0, 5, (256,)).float()}


def test_emacallback_trains_and_tracks():
    model = build_samepad_chrombpnet(out_window=256, n_filters=8, n_layers=2)
    lit = ChromBPNetLit(model, alpha=0.8, beta=1.0, lr=1e-3)
    ema_cb = EMACallback(decay=0.9)
    trainer = L.Trainer(
        max_steps=4,
        limit_train_batches=4,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[ema_cb],
    )
    trainer.fit(lit, DataLoader(_ToyDS(8), batch_size=4))
    assert ema_cb.ema is not None and len(ema_cb.ema.shadow) > 0
    # EMA weights differ from the live weights after a few updates.
    diff = sum(
        (ema_cb.ema.shadow[n] - p).abs().sum().item()
        for n, p in lit.named_parameters()
        if n in ema_cb.ema.shadow
    )
    assert diff > 0
