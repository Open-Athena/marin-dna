"""Online QTL eval: window extraction, log2FC scoring, and the live callback."""

import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from marin_dna.pipelines.chrombpnet_eval.lit import ChromBPNetLit
from marin_dna.pipelines.chrombpnet_eval.onehot import build_onehot_chrombpnet
from marin_dna.pipelines.chrombpnet_eval.qtl_eval import (
    QTLEvalCallback,
    QTLSpec,
    extract_ref_alt_onehot,
    score_log2fc,
    signed_pearson,
)

HALF = 2114 // 2  # variant sits at the center index


def _genome_all_a(chrom: str, start: int, end: int, strand: str = "+") -> str:
    # A fake genome that returns all-'A' of the requested length.
    return "A" * (end - start)


def _ref_alt_onehot(n: int) -> tuple[np.ndarray, np.ndarray]:
    ref = np.zeros((n, 4, 2114), dtype=np.float32)
    ref[:, 0, :] = 1.0  # all A
    alt = ref.copy()
    alt[:, 0, HALF] = 0.0
    alt[:, 1, HALF] = 1.0  # center -> C
    return ref, alt


def test_extract_ref_alt_onehot_basic():
    ref_oh, alt_oh = extract_ref_alt_onehot(
        ["1", "2", "X"],
        [2000, 3000, 5000],
        ["A", "A", "A"],
        ["C", "G", "T"],
        _genome_all_a,
    )
    assert ref_oh.shape == (3, 4, 2114) and alt_oh.shape == (3, 4, 2114)
    # ref center base is A (channel 0); no other channel set at the center
    assert ref_oh[:, 0, HALF].all() and ref_oh[:, 1:, HALF].sum() == 0
    # alt center is the substituted allele (C=1, G=2, T=3)
    assert (
        alt_oh[0, 1, HALF] == 1 and alt_oh[1, 2, HALF] == 1 and alt_oh[2, 3, HALF] == 1
    )
    # flanks untouched (still A)
    assert alt_oh[0, 0, 0] == 1 and alt_oh[0, 0, HALF - 1] == 1


def test_extract_ref_mismatch_raises():
    # genome is all-A but we claim ref=C everywhere -> 0% match -> loud failure.
    with pytest.raises(AssertionError, match="reference-build"):
        extract_ref_alt_onehot(["1"], [2000], ["C"], ["G"], _genome_all_a)


def test_extract_rejects_indel():
    with pytest.raises(AssertionError, match="SNP-only"):
        extract_ref_alt_onehot(["1"], [2000], ["A"], ["AC"], _genome_all_a)


def test_score_log2fc_shape_and_finite():
    model = build_onehot_chrombpnet(bias_h5=None, n_filters=8, n_layers=2)
    ref, alt = _ref_alt_onehot(5)
    scores = score_log2fc(model, ref, alt, batch_size=2)
    assert scores.shape == (5,) and np.isfinite(scores).all()
    # identical ref/alt -> zero effect
    z = score_log2fc(model, ref, ref, batch_size=2)
    assert np.allclose(z, 0.0, atol=1e-4)


def test_signed_pearson_constant_is_zero():
    assert signed_pearson(np.ones(5), np.arange(5.0)) == 0.0


class _ToyDS(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, i: int) -> dict:
        oh = torch.zeros(4, 2114)
        oh[torch.randint(0, 4, (2114,)), torch.arange(2114)] = 1.0
        return {"onehot_seq": oh, "profile": torch.randint(0, 5, (1000,)).float()}


def test_callback_logs_qtl_metrics():
    L.seed_everything(0)  # deterministic init+training (no flaky model divergence)
    model = build_onehot_chrombpnet(bias_h5=None, n_filters=8, n_layers=2)
    lit = ChromBPNetLit(model, alpha=1.0, beta=1.0, lr=1e-3)
    ref, alt = _ref_alt_onehot(6)
    spec = QTLSpec("caqtl", ref, alt, np.random.default_rng(0).standard_normal(6))
    trainer = L.Trainer(
        max_steps=2,
        limit_train_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[QTLEvalCallback([spec], batch_size=4, every_n_steps=1)],
    )
    trainer.fit(lit, DataLoader(_ToyDS(), batch_size=4))  # no val loop (#259)
    metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
    assert "qtl_caqtl_pearson" in metrics, sorted(metrics)
    assert "qtl_avg_pearson" in metrics, sorted(metrics)
    assert torch.isfinite(torch.as_tensor(float(metrics["qtl_caqtl_pearson"])))
