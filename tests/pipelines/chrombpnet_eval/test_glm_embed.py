"""Smoke tests for the frozen-gLM ChromBPNet arm (#243 M2).

Synthetic, CPU, no checkpoint download: a tiny **stub gLM** (an ``nn.Module`` whose
``forward(input_ids)`` returns ``.last_hidden_state``) stands in for the real HF
model, so these cover the embedding front-end (one-hot→ids, BOS, FWD‖RC concat +
re-alignment), the same-pad tower shapes, the frozen-gLM gradient flow, the WSD
training loop, and the QTL scoring path — without GPU or network.
"""

from types import SimpleNamespace

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from marin_dna.pipelines.chrombpnet_eval.glm_embed import build_glm_samepad_chrombpnet
from marin_dna.pipelines.chrombpnet_eval.glm_lit import (
    GLMChromBPNetLit,
    signed_pearson,
    wsd_lr_lambda,
)
from marin_dna.pipelines.chrombpnet_eval.qtl_eval import score_log2fc

HIDDEN = 8
L_WIN = 16
VOCAB = 7  # matches exp136 (pad/unk/bos + acgt)


class _StubGLM(nn.Module):
    """Deterministic per-token stand-in for the HF gLM: ``forward(input_ids)`` →
    ``SimpleNamespace(last_hidden_state=[B,T,HIDDEN])``."""

    def __init__(self, vocab_size: int = VOCAB, hidden: int = HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.mix = nn.Linear(hidden, hidden, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.mix(self.embed(input_ids)))


def _build(
    rc: bool = True,
    emb_norm: bool = True,
    out_window: int = L_WIN,
    proj_dim: int | None = 8,
):
    return build_glm_samepad_chrombpnet(
        _StubGLM(),
        hidden_size=HIDDEN,
        out_window=out_window,
        proj_dim=proj_dim,
        rc=rc,
        emb_norm=emb_norm,
        n_filters=8,
        n_layers=2,
    )


def _rand_onehot(b: int = 3, length: int = L_WIN) -> torch.Tensor:
    oh = torch.zeros(b, 4, length)
    idx = torch.randint(0, 4, (b, length))
    oh.scatter_(1, idx.unsqueeze(1), 1.0)
    return oh


def test_forward_shapes():
    model = _build()
    profile, count = model(_rand_onehot())
    assert profile.shape == (3, L_WIN), profile.shape
    assert count.shape == (3, 1), count.shape
    assert torch.isfinite(profile).all() and torch.isfinite(count).all()


def test_embedding_dim_doubles_with_rc():
    assert _build(rc=True).embedding_dim == 2 * HIDDEN
    assert _build(rc=False).embedding_dim == HIDDEN
    # rc=False forward still satisfies the contract.
    profile, count = _build(rc=False)(_rand_onehot())
    assert profile.shape == (3, L_WIN) and count.shape == (3, 1)


def test_pointwise_proj_reduces_head_params():
    """The pointwise (1×1) FWD‖RC fusion before the tower has far fewer params than
    letting the wide k21 iconv ingest the full 2H concat — and both still satisfy
    the ChromBPNet forward contract."""
    proj = _build(proj_dim=8)
    concat = _build(proj_dim=None)
    assert proj.proj is not None and concat.proj is None

    def n_trainable(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    assert n_trainable(proj) < n_trainable(concat)
    for m in (proj, concat):
        pr, ct = m(_rand_onehot())
        assert pr.shape == (3, L_WIN) and ct.shape == (3, 1)


def test_fwd_rc_equivariance():
    """``_embed(rc(x))`` must equal the length-flip + half-swap of ``_embed(x)``:
    re-aligning the RC strand and concatenating is correct iff feeding the RC input
    just swaps the two halves and reverses position order. Catches a wrong flip or
    a wrong channel-complement."""
    model = _build(emb_norm=False).eval()
    x = _rand_onehot()
    rc_x = x[:, [3, 2, 1, 0], :].flip(-1)
    with torch.no_grad():
        emb = model._embed(x)  # [B, L, 2H]
        emb_rc = model._embed(rc_x)  # [B, L, 2H]
    fwd_half, rc_half = emb[..., :HIDDEN], emb[..., HIDDEN:]
    expected = torch.cat([rc_half, fwd_half], dim=-1).flip(1)  # swap halves + flip L
    assert torch.allclose(emb_rc, expected, atol=1e-5), (
        (emb_rc - expected).abs().max().item()
    )


def test_glm_is_frozen_and_only_head_trains():
    model = _build()
    assert all(not p.requires_grad for p in model.glm.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("glm.")]
    assert head and all(p.requires_grad for p in head)
    # train() must keep the gLM in eval (frozen encoder), even after model.train().
    model.train()
    assert not model.glm.training
    # A backward leaves gLM grads None; head grads populated.
    profile, count = model(_rand_onehot())
    (profile.sum() + count.sum()).backward()
    assert all(p.grad is None for p in model.glm.parameters())
    assert any(p.grad is not None for p in head)


class _ToyDS(Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, L_WIN)
        oh[torch.randint(0, 4, (L_WIN,)), torch.arange(L_WIN)] = 1.0
        profile = torch.randint(0, 5, (L_WIN,)).float()
        return {"onehot_seq": oh, "profile": profile}


def test_wsd_training_loop_runs():
    model = _build()
    lit = GLMChromBPNetLit(
        model, alpha=1.0, beta=1.0, lr=1e-3, optimizer="adamw", weight_decay=0.01
    )
    trainer = L.Trainer(
        max_steps=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, DataLoader(_ToyDS(8), batch_size=4))
    # WSD scheduler configured on a step interval; grad_norm hook fired finite/pos.
    assert trainer.lr_scheduler_configs[0].interval == "step"
    metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
    gn = float(metrics["grad_norm"])
    assert gn > 0 and np.isfinite(gn), gn


def test_score_log2fc_runs_on_embedding_model():
    """The QTL scorer (calls ``model(onehot)``) works unchanged on the embedding
    model — the callback path, exercised without any HF download."""
    model = _build().eval()
    n = 5
    ref = np.zeros((n, 4, L_WIN), dtype=np.float32)
    ref[:, np.random.randint(0, 4, L_WIN), np.arange(L_WIN)] = 1.0
    alt = ref.copy()
    alt[:, :, L_WIN // 2] = np.eye(4, dtype=np.float32)[np.random.randint(0, 4, n)]
    scores = score_log2fc(model, ref, alt, batch_size=2)
    assert scores.shape == (n,) and np.isfinite(scores).all()


def test_wsd_lr_lambda_shape():
    f = wsd_lr_lambda(total_steps=100, warmup_frac=0.1, decay_frac=0.2)
    assert f(0) < f(5) < f(10)  # warmup ramps up
    assert f(10) == 1.0 and f(50) == 1.0  # stable phase at peak
    assert f(80) == 1.0 and f(90) < 1.0 and f(99) < f(90)  # decay 1 → 0
    assert f(100) == 0.0


def test_signed_pearson_degenerate_is_zero():
    assert signed_pearson(np.array([1.0]), np.array([1.0])) == 0.0  # < 2 points
    assert signed_pearson(np.ones(5), np.arange(5.0)) == 0.0  # constant scores
    r = signed_pearson(np.arange(5.0), np.arange(5.0))
    assert abs(r - 1.0) < 1e-9  # perfectly correlated
