"""PyTorch dataset for enhancer classification from parquet splits."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from torch.utils.data import Dataset


class EnhancerDataset(Dataset):
    """Binary enhancer classification dataset backed by a parquet file.

    Each item is a (one_hot, label) pair where one_hot is a float32 tensor
    of shape (seq_len, 4) and label is a scalar float32 tensor.

    Args:
        parquet_path: Path to a parquet file with columns ``seq`` and ``label``.
    """

    def __init__(self, parquet_path: str | Path) -> None:
        df = pl.read_parquet(parquet_path, columns=["seq", "label"])
        self.sequences: list[str] = df["seq"].to_list()
        self.labels: np.ndarray = df["label"].to_numpy().astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        onehot = sequence_to_onehot(self.sequences[idx]).astype(np.float32)
        return (
            torch.from_numpy(onehot),
            torch.tensor(self.labels[idx]),
        )
