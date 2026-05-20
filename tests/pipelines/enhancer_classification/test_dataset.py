"""Tests for EnhancerDataset."""

import tempfile
from pathlib import Path

import polars as pl
import torch

from marin_dna.pipelines.enhancer_classification.dataset import EnhancerDataset

SEQ_LEN = 255


def _make_parquet(path: Path, n: int = 10) -> None:
    """Write a minimal synthetic parquet with random DNA sequences."""
    import random

    random.seed(0)
    seqs = ["".join(random.choices("ACGT", k=SEQ_LEN)) for _ in range(n)]
    labels = [i % 2 for i in range(n)]
    pl.DataFrame({"seq": seqs, "label": labels}).write_parquet(path)


def test_dataset_length():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        _make_parquet(path, n=8)
        ds = EnhancerDataset(path)
        assert len(ds) == 8


def test_dataset_shapes_and_dtypes():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        _make_parquet(path, n=4)
        ds = EnhancerDataset(path)
        x, y = ds[0]
        assert x.shape == (SEQ_LEN, 4)
        assert x.dtype == torch.float32
        assert y.shape == ()
        assert y.dtype == torch.float32


def test_dataset_label_values():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        _make_parquet(path, n=6)
        ds = EnhancerDataset(path)
        labels = [ds[i][1].item() for i in range(len(ds))]
        assert set(labels) == {0.0, 1.0}


def test_dataset_onehot_valid():
    """Each row of the one-hot should sum to exactly 1 (no N bases in synthetic data)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        _make_parquet(path, n=2)
        ds = EnhancerDataset(path)
        x, _ = ds[0]
        row_sums = x.sum(dim=1)
        assert torch.all(row_sums == 1.0)
