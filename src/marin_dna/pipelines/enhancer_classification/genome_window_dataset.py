"""Shared dataset for window-based model inference over a genome."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from torch.utils.data import Dataset

from marin_dna.pipelines.enhancer_classification.genome import Genome


class GenomeWindowDataset(Dataset):
    """Map-style dataset of genomic windows for model inference.

    Each DataLoader worker opens its own ``py2bit`` / FASTA handle on first
    access. For 2bit files this means shared memory-mapped pages across
    workers; the full genome is never materialized as one-hot tensors.

    Args:
        genome_path: Path to genome file (.2bit or .fa/.fa.gz).
        windows: DataFrame with ``chrom``, ``start``, ``end`` columns.
    """

    def __init__(self, genome_path: Path, windows: pl.DataFrame) -> None:
        self._genome_path = genome_path
        self._chroms = windows["chrom"].to_numpy().astype(str)
        self._starts = windows["start"].to_numpy().astype(np.int64)
        self._ends = windows["end"].to_numpy().astype(np.int64)
        self._genome: Genome | None = None

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._genome is None:
            self._genome = Genome(self._genome_path)
        seq = self._genome(
            self._chroms[idx], int(self._starts[idx]), int(self._ends[idx])
        )
        onehot = sequence_to_onehot(seq).astype(np.float32)
        return torch.from_numpy(onehot)
