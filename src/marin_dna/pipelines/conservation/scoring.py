"""Per-window phyloP scoring via pyBigWig.

Performance: pyBigWig handles ~10K windows/sec/core on 255 bp windows;
parallelise across chroms in the calling pipeline.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig


def _bw_chrom(chrom: str, prefix: str = "chr") -> str:
    """Add ``prefix`` to ``chrom`` unless it's already there.

    Bridges Ensembl bare chrom names (``"1"``) to the ``"chr1"`` style
    that UCSC bigWigs use.
    """
    return chrom if chrom.startswith(prefix) else f"{prefix}{chrom}"


def score_windows(
    bw_path: str | Path,
    windows: pl.DataFrame,
    threshold: float,
    *,
    chrom_prefix: str = "chr",
) -> pl.DataFrame:
    """Score each window in ``windows`` against the bigWig at ``bw_path``.

    Args:
        bw_path: path to the phyloP bigWig (UCSC ``chr1``-style chrom names).
        windows: polars frame with columns ``chrom``, ``start``, ``end``;
            ``chrom`` may be either ``"1"`` or ``"chr1"``-style — bare names
            are auto-prefixed with ``chrom_prefix`` for the bigWig lookup.
        threshold: phyloP value at/above which a base is "conserved".
        chrom_prefix: prefix to add to bare chrom names (default ``"chr"``).

    Returns:
        ``windows`` with four extra columns:
          - ``conserved_bases`` (Int32): count of bases with value ≥
            ``threshold``. NaN positions are counted as **non-conserved**
            (= 0), so the count is over the whole window length, not just
            the covered (non-NaN) bases.
          - ``proportion_conserved`` (Float32): ``conserved_bases / (end - start)``.
          - ``mean_phylop`` (Float32): mean of finite values in the window
            (NaN if no finite values).
          - ``n_valid_bases`` (Int32): count of bases where the bigWig has
            a finite value (i.e. excludes NaN).
    """
    assert {"chrom", "start", "end"}.issubset(windows.columns), (
        f"windows must have chrom/start/end columns; got {windows.columns}"
    )

    conserved_bases = np.empty(len(windows), dtype=np.int32)
    proportion_conserved = np.empty(len(windows), dtype=np.float32)
    mean_phylop = np.empty(len(windows), dtype=np.float32)
    n_valid_bases = np.empty(len(windows), dtype=np.int32)

    bw = pyBigWig.open(str(bw_path))
    try:
        bw_chroms = set(bw.chroms())
        chroms = windows["chrom"].to_list()
        starts = windows["start"].to_list()
        ends = windows["end"].to_list()
        for i, (chrom, start, end) in enumerate(zip(chroms, starts, ends)):
            start = int(start)
            end = int(end)
            size = end - start
            assert size > 0, f"empty window at index {i}: {chrom}:{start}-{end}"
            bw_chrom = _bw_chrom(chrom, chrom_prefix)
            if bw_chrom not in bw_chroms:
                conserved_bases[i] = 0
                proportion_conserved[i] = 0.0
                mean_phylop[i] = np.nan
                n_valid_bases[i] = 0
                continue
            values = np.asarray(
                bw.values(bw_chrom, start, end, numpy=True), dtype=np.float64
            )
            finite_mask = np.isfinite(values)
            n_finite = int(finite_mask.sum())
            # NaN treated as non-conserved: NaN >= threshold evaluates to
            # False in NumPy regardless of threshold, so the NaN positions
            # are naturally excluded from the conserved-bases count
            # *without* needing to fill them. (Don't fill with 0 — that
            # breaks the case where threshold <= 0.)
            n_conserved = int((values >= threshold).sum())
            conserved_bases[i] = n_conserved
            proportion_conserved[i] = n_conserved / size
            mean_phylop[i] = float(np.nanmean(values)) if n_finite > 0 else np.nan
            n_valid_bases[i] = n_finite
    finally:
        bw.close()

    return windows.with_columns(
        [
            pl.Series("conserved_bases", conserved_bases, dtype=pl.Int32),
            pl.Series("proportion_conserved", proportion_conserved, dtype=pl.Float32),
            pl.Series("mean_phylop", mean_phylop, dtype=pl.Float32),
            pl.Series("n_valid_bases", n_valid_bases, dtype=pl.Int32),
        ]
    )
