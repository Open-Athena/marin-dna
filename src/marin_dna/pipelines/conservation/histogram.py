"""Per-base phyloP-value histograms over defined genome regions.

These histograms are the input to ``calibration.calibrate_to_match_count``:
given the histogram for the reference track (e.g. ``phyloP_241m``) at a
reference threshold, find the threshold for a target track (e.g.
``phyloP_447m``) whose genome-wide passing-nucleotide count matches.

Bases with no bigWig signal are counted in ``n_nan`` (separately from the
``counts`` array). Values outside ``[edges[0], edges[-1]]`` go into the
first/last bin (``np.clip`` semantics) — pick a wide-enough range that this
clipping is negligible. For phyloP-mammal tracks ``[-20, 20]`` covers the
distribution with margin.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from .scoring import _bw_chrom


@dataclass
class PhylopHistogram:
    """Histogram of per-base phyloP values plus a separate NaN count.

    ``counts`` is a length-``n_bins`` integer array; ``edges`` is the
    length-``n_bins + 1`` float array of bin edges. Bin i covers
    ``[edges[i], edges[i + 1])``.
    """

    counts: np.ndarray
    edges: np.ndarray
    n_nan: int

    def __post_init__(self) -> None:
        assert self.counts.ndim == 1
        assert self.edges.ndim == 1
        assert len(self.edges) == len(self.counts) + 1, (
            f"edges length must be counts length + 1, "
            f"got {len(self.edges)} vs {len(self.counts)}"
        )
        assert np.all(np.diff(self.edges) > 0), "edges must be strictly increasing"
        assert (self.counts >= 0).all(), "counts must be non-negative"
        assert self.n_nan >= 0

    @property
    def n_bins(self) -> int:
        return int(len(self.counts))

    def total(self) -> int:
        """Total non-NaN bases counted."""
        return int(self.counts.sum())

    def total_with_nan(self) -> int:
        """Total bases counted, including NaN."""
        return self.total() + self.n_nan

    def count_ge(self, threshold: float) -> int:
        """Count of non-NaN bases with value >= ``threshold``.

        Within the bracketing bin we linearly interpolate to handle a
        threshold that falls between bin edges. Bins fully above
        ``threshold`` contribute their full count.
        """
        if threshold <= self.edges[0]:
            return self.total()
        if threshold >= self.edges[-1]:
            return 0

        i = int(np.searchsorted(self.edges, threshold, side="right")) - 1
        # i is the index of the bin containing `threshold`.
        # Linear interpolation within that bin: assume values uniform on [edges[i], edges[i+1]).
        bin_lo = self.edges[i]
        bin_hi = self.edges[i + 1]
        frac_above = (bin_hi - threshold) / (bin_hi - bin_lo)
        partial = int(round(self.counts[i] * frac_above))
        full = int(self.counts[i + 1 :].sum())
        return partial + full

    def threshold_for_count(self, target_count: int) -> float:
        """Find threshold T such that ``count_ge(T) ≈ target_count``.

        Returns the linearly-interpolated threshold within the bracketing
        bin. Monotone-decreasing inverse of ``count_ge``.
        """
        if target_count <= 0:
            return float(self.edges[-1])
        total = self.total()
        if target_count >= total:
            return float(self.edges[0])

        # Reverse cumsum from the top: cum[i] = count of bases in bins [i, n_bins).
        rev_cum = np.concatenate(
            [np.cumsum(self.counts[::-1])[::-1], np.array([0])]
        )  # length n_bins + 1; rev_cum[i] = sum of counts[i:]
        # Find smallest i with rev_cum[i] >= target_count.
        i = int(np.searchsorted(-rev_cum, -target_count))
        # rev_cum is non-increasing, so searchsorted on the negated array gives
        # the first position where rev_cum drops below target_count.
        # Safety: the i we want is one less (the bin we're inside).
        i = max(0, i - 1)
        # Now bin i contains the threshold. rev_cum[i] >= target_count, rev_cum[i+1] < target_count.
        bin_lo = self.edges[i]
        bin_hi = self.edges[i + 1]
        in_bin = self.counts[i]
        if in_bin == 0:
            return float(bin_hi)
        # Need (rev_cum[i + 1] + frac * in_bin) == target_count, where `frac` is the
        # fraction of the bin above the threshold; threshold = bin_hi - frac * (bin_hi - bin_lo).
        needed_in_bin = target_count - rev_cum[i + 1]
        frac = needed_in_bin / in_bin
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(bin_hi - frac * (bin_hi - bin_lo))

    def __add__(self, other: "PhylopHistogram") -> "PhylopHistogram":
        assert np.array_equal(self.edges, other.edges), (
            "Histograms must share the same edges to be added."
        )
        return PhylopHistogram(
            counts=self.counts + other.counts,
            edges=self.edges.copy(),
            n_nan=self.n_nan + other.n_nan,
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            counts=self.counts,
            edges=self.edges,
            n_nan=np.array([self.n_nan], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PhylopHistogram":
        d = np.load(path)
        return cls(
            counts=d["counts"].astype(np.int64),
            edges=d["edges"].astype(np.float64),
            n_nan=int(d["n_nan"][0]),
        )


def build_histogram_for_chrom(
    bw_path: str | Path,
    chrom: str,
    defined_intervals: pd.DataFrame,
    edges: np.ndarray,
    chrom_prefix: str = "chr",
) -> PhylopHistogram:
    """Build a histogram of phyloP values for ``chrom`` over ``defined_intervals``.

    ``defined_intervals`` is a 0-based half-open ``[chrom, start, end]`` frame
    restricted to a single chromosome — typically the per-chrom slice of a
    genome-wide "defined regions" BED (i.e. ACGT, no N runs). The bigWig is
    expected to use UCSC-style ``chr1`` chromosome names; if ``chrom`` lacks
    that prefix, ``chrom_prefix`` is prepended.

    Returns a ``PhylopHistogram`` covering only the bases inside
    ``defined_intervals``. Asserts that the total accounted bases matches
    the sum of interval lengths (``hist.total() + hist.n_nan ==
    sum(end - start)``).
    """
    assert defined_intervals["chrom"].nunique() == 1, (
        "defined_intervals must contain exactly one chromosome"
    )
    assert defined_intervals["chrom"].iloc[0] == chrom, (
        f"chrom mismatch: dataframe has {defined_intervals['chrom'].iloc[0]!r}, "
        f"expected {chrom!r}"
    )
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    n_nan = 0
    expected_total = 0

    bw_chrom = _bw_chrom(chrom, chrom_prefix)
    bw = pyBigWig.open(str(bw_path))
    try:
        chroms_in_bw = bw.chroms()
        assert bw_chrom in chroms_in_bw, (
            f"bigWig {bw_path} has no chrom {bw_chrom!r}; "
            f"available example: {next(iter(chroms_in_bw))}"
        )
        for _, row in defined_intervals.iterrows():
            start, end = int(row["start"]), int(row["end"])
            assert start < end, f"empty/inverted interval: {chrom}:{start}-{end}"
            expected_total += end - start
            values = np.asarray(
                bw.values(bw_chrom, start, end, numpy=True), dtype=np.float64
            )
            nan_mask = np.isnan(values)
            n_nan += int(nan_mask.sum())
            finite = values[~nan_mask]
            if finite.size:
                hist, _ = np.histogram(np.clip(finite, edges[0], edges[-1]), bins=edges)
                counts += hist.astype(np.int64)
    finally:
        bw.close()

    h = PhylopHistogram(
        counts=counts, edges=np.asarray(edges, dtype=np.float64), n_nan=n_nan
    )
    assert h.total() + h.n_nan == expected_total, (
        f"histogram total ({h.total()}) + n_nan ({h.n_nan}) "
        f"!= expected bases ({expected_total}) for {chrom}"
    )
    return h
