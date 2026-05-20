"""PyTorch dataset for per-bin enhancer segmentation."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from Bio.Seq import Seq
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Per-bin segmentation dataset backed by a parquet file.

    Each item is a ``(one_hot, labels, genome_idx)`` triple where ``one_hot``
    is a float32 tensor of shape ``(window_size, 4)``, ``labels`` is a
    float32 tensor of shape ``(num_bins,)``, and ``genome_idx`` is a scalar
    long tensor identifying the species (index into ``self.genomes``).

    With ``augment_rc=True`` the dataset's virtual length is doubled: index
    ``i < N`` returns the forward sample, index ``i >= N`` returns the
    reverse-complement of sample ``i - N`` with its per-bin labels reversed.
    This matches pre-baking ``forward + RC`` into the parquet but avoids the
    2x memory spike at dataset-build time.

    Args:
        parquet_path: Path to a parquet file with columns ``seq`` (str),
            ``labels`` (list[int] / list[uint8]), and ``genome`` (str).
        augment_rc: If True (train split), double virtual length with RC
            samples. Default False (val split).
        genome_to_idx: Optional mapping from genome name to integer index.
            If None, built from the unique genomes in this parquet (sorted
            alphabetically). Pass the same mapping to train and val datasets
            so the per-species metrics align.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        augment_rc: bool = False,
        genome_to_idx: dict[str, int] | None = None,
    ) -> None:
        df = pl.read_parquet(parquet_path, columns=["seq", "labels", "genome"])
        self.sequences: list[str] = df["seq"].to_list()
        # Store labels as a single (N, num_bins) uint8 array for fast indexing.
        self.labels: np.ndarray = np.asarray(df["labels"].to_list(), dtype=np.uint8)

        genomes_col = df["genome"].to_list()
        if genome_to_idx is None:
            genome_to_idx = {g: i for i, g in enumerate(sorted(set(genomes_col)))}
        self.genome_to_idx = genome_to_idx
        self.genome_idx: np.ndarray = np.asarray(
            [genome_to_idx[g] for g in genomes_col], dtype=np.int64
        )

        self.augment_rc = augment_rc
        self._n = len(self.sequences)

    @property
    def genomes(self) -> list[str]:
        return [g for g, _ in sorted(self.genome_to_idx.items(), key=lambda kv: kv[1])]

    def __len__(self) -> int:
        return 2 * self._n if self.augment_rc else self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.augment_rc and idx >= self._n:
            base = idx - self._n
            seq = str(Seq(self.sequences[base]).reverse_complement())
            labels = self.labels[base][::-1].astype(np.float32)
            g = int(self.genome_idx[base])
        else:
            seq = self.sequences[idx]
            labels = self.labels[idx].astype(np.float32)
            g = int(self.genome_idx[idx])
        onehot = sequence_to_onehot(seq).astype(np.float32)
        return (
            torch.from_numpy(onehot),
            torch.from_numpy(np.ascontiguousarray(labels)),
            torch.tensor(g, dtype=torch.long),
        )
