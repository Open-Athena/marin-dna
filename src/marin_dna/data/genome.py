"""Random-access genome sequence reader (FASTA, pyfaidx-backed).

Vendored from biofoundation/data.py at commit 834dd4c (May 2026), simplified
to drop the GenomicSet class and transform helpers — those live in
marin-dna's own ``intervals.py`` and ``transforms.py`` respectively.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pyfaidx import Fasta

from marin_dna.data.dna import reverse_complement


class Genome:
    """Random-access FASTA reader backed by :mod:`pyfaidx`.

    Sequences are read on demand from the FASTA file rather than loaded into
    memory upfront, so ``Genome(...)`` is fast to construct and uses
    near-zero baseline memory.

    A samtools-compatible ``.fai`` index is required. For local files, pyfaidx
    creates one next to the FASTA on first open. For remote paths
    (e.g. ``s3://``), the ``.fai`` must already exist alongside the FASTA.

    Gzipped FASTA input must be **bgzipped** (BGZF, with a ``.gzi``
    companion). Plain gzip is not supported — re-compress with ``bgzip``.

    Remote paths (``s3://``, etc.) require ``fsspec`` and ``s3fs`` to be
    installed; marin-dna doesn't ship them as direct deps — install them
    explicitly if needed.

    Args:
        path: Local filesystem path or fsspec-compatible URL.
        subset_chroms: If given, only chromosomes in this set are exposed.
        storage_options: Forwarded to ``fsspec.open`` for remote paths
            (e.g. ``{"anon": True}`` for public S3 buckets). Ignored for
            local paths.
    """

    def __init__(
        self,
        path: str | Path,
        subset_chroms: set[str] | None = None,
        storage_options: dict[str, Any] | None = None,
    ):
        self._path: str = str(path)
        self._is_remote: bool = urlparse(self._path).scheme not in ("", "file")
        self._storage_options: dict[str, Any] = dict(storage_options or {})
        self._subset_chroms: set[str] | None = (
            set(subset_chroms) if subset_chroms is not None else None
        )

        # Defer the chrom-size probe to first use. If we probed eagerly here,
        # remote (s3://, etc.) paths would initialize fsspec's async event
        # loop in the parent process — and that loop reference doesn't
        # survive ``fork()`` into PyTorch DataLoader workers, deadlocking
        # any S3 read in the worker. Lazy probing means each fork()'d
        # worker triggers a fresh fsspec init on its own first ``__call__``.
        self._chrom_sizes: dict[str, int] | None = None

        self._fa: Fasta | None = None
        self._fa_pid: int = -1

    def _ensure_probed(self) -> None:
        if self._chrom_sizes is not None:
            return
        with self._open_fasta() as fa:
            # Unlike a normal mapping, iterating pyfaidx.Fasta yields FastaRecord objects.
            fa_keys = fa.keys()
            keys = [
                k
                for k in fa_keys
                if self._subset_chroms is None or k in self._subset_chroms
            ]
            self._chrom_sizes = {k: len(fa[k]) for k in keys}

    def _open_fasta(self) -> Fasta:
        if self._is_remote:
            import fsspec

            return Fasta(fsspec.open(self._path, **self._storage_options), as_raw=True)
        return Fasta(self._path, as_raw=True)

    def _fasta(self) -> Fasta:
        pid = os.getpid()
        if self._fa is None or self._fa_pid != pid:
            self._fa = self._open_fasta()
            self._fa_pid = pid
        return self._fa

    @property
    def chroms(self) -> dict[str, int]:
        """Chromosome names mapped to their lengths."""
        self._ensure_probed()
        assert self._chrom_sizes is not None
        return dict(self._chrom_sizes)

    def __call__(
        self,
        chrom: str,
        start: int,
        end: int,
        strand: Literal["+", "-"] = "+",
    ) -> str:
        """Get a subsequence of a chromosome.

        If start is negative, the sequence is padded with Ns on the left.
        If end is greater than the chromosome size, the sequence is padded
        with Ns on the right.

        Args:
            chrom: The chromosome to get the sequence of.
            start: The start position of the sequence (0-based, inclusive).
            end: The end position of the sequence (0-based, exclusive).
            strand: The strand of the sequence (+ or -).
        """
        self._ensure_probed()
        assert self._chrom_sizes is not None
        if chrom not in self._chrom_sizes:
            raise ValueError(f"chromosome {chrom} not found in genome")
        chrom_size = self._chrom_sizes[chrom]
        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        if start > end:
            raise ValueError(f"start {start} must be less than or equal to end {end}")
        if end < 0:
            raise ValueError(f"end {end} must be non-negative for chromosome {chrom}")
        if start >= chrom_size:
            raise ValueError(f"start {start} is out of range for chromosome {chrom}")

        seq: str = self._fasta()[chrom][max(start, 0) : min(end, chrom_size)]

        if start < 0:
            seq = "N" * (-start) + seq  # left padding
        if end > chrom_size:
            seq = seq + "N" * (end - chrom_size)  # right padding

        if strand == "-":
            seq = reverse_complement(seq)
        return seq

    def __getstate__(self) -> dict[str, Any]:
        # Don't pickle the live Fasta handle: its fd is invalid in a spawn worker.
        state = self.__dict__.copy()
        state["_fa"] = None
        state["_fa_pid"] = -1
        return state
