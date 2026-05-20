"""Genome sequence reader supporting FASTA and 2bit formats.

Provides a unified ``Genome`` interface for extracting DNA subsequences
by genomic coordinates. The 2bit backend uses memory-mapped random access
via py2bit, making it suitable for multi-worker DataLoader usage where
each worker opens its own file handle backed by shared OS page cache.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Literal

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


class Genome:
    """Read-only genome sequence accessor.

    Supports both FASTA (.fa, .fa.gz) and 2bit (.2bit) formats. The 2bit
    backend avoids loading the entire genome into memory.

    Args:
        path: Path to genome file (.fa, .fa.gz, or .2bit).
        subset_chroms: If given, only load these chromosomes (FASTA only;
            ignored for 2bit which always has random access).
    """

    def __init__(
        self,
        path: str | Path,
        subset_chroms: set[str] | None = None,
    ) -> None:
        self._path = Path(path)
        name = self._path.name

        if name.endswith(".2bit"):
            import py2bit

            self._tb = py2bit.open(str(self._path))
            self._chrom_sizes: dict[str, int] = self._tb.chroms()
            self._genome: pd.Series | None = None
        else:
            self._tb = None
            open_fn = (
                gzip.open if name.endswith(".gz") else open  # noqa: SIM115
            )
            with open_fn(str(self._path), "rt") as handle:
                self._genome = pd.Series(
                    {
                        rec.id: str(rec.seq)
                        for rec in SeqIO.parse(handle, "fasta")
                        if subset_chroms is None or rec.id in subset_chroms
                    }
                )
            self._chrom_sizes = {chrom: len(seq) for chrom, seq in self._genome.items()}

    @property
    def chroms(self) -> dict[str, int]:
        """Chromosome names mapped to their lengths."""
        return dict(self._chrom_sizes)

    def __call__(
        self,
        chrom: str,
        start: int,
        end: int,
        strand: Literal["+", "-"] = "+",
    ) -> str:
        """Extract a subsequence from the genome.

        Out-of-bounds coordinates are padded with ``N``. Negative start
        values pad on the left; end values beyond the chromosome length
        pad on the right.

        Args:
            chrom: Chromosome name.
            start: Start position (0-based, inclusive).
            end: End position (0-based, exclusive).
            strand: ``"+"`` for forward, ``"-"`` for reverse complement.

        Returns:
            Uppercase DNA string of length ``end - start``.
        """
        if chrom not in self._chrom_sizes:
            raise ValueError(f"chromosome {chrom} not found in genome")
        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        if start > end:
            raise ValueError(f"start {start} must be <= end {end}")

        chrom_size = self._chrom_sizes[chrom]

        if start >= chrom_size:
            raise ValueError(
                f"start {start} is out of range for chromosome {chrom} (size {chrom_size})"
            )
        if end < 0:
            raise ValueError(f"end {end} must be non-negative for chromosome {chrom}")

        clamped_start = max(start, 0)
        clamped_end = min(end, chrom_size)

        if self._tb is not None:
            seq = self._tb.sequence(chrom, clamped_start, clamped_end)
        else:
            assert self._genome is not None
            seq = self._genome[chrom][clamped_start:clamped_end]

        seq = seq.upper()

        if start < 0:
            seq = "N" * (-start) + seq
        if end > chrom_size:
            seq = seq + "N" * (end - chrom_size)

        if strand == "-":
            seq = str(Seq(seq).reverse_complement())

        return seq
