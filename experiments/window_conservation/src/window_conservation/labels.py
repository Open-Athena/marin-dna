"""Reuse the projection pipeline's phyloP >= 2.2162 whole-window definition."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pyBigWig


def summarize(values: np.ndarray, threshold: float) -> tuple[int, int, float]:
    finite = np.isfinite(values)
    assert not np.isinf(values).any()
    return (
        int((values >= threshold).sum()),
        int(finite.sum()),
        float(np.mean(values[finite])) if finite.any() else float("nan"),
    )


def window_labels(
    path: Path, chrom: str, starts: np.ndarray, ends: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    conserved = np.zeros(len(starts), dtype=np.int32)
    coverage = np.zeros(len(starts), dtype=np.int32)
    means = np.full(len(starts), np.nan)
    with pyBigWig.open(str(path)) as bw:
        chroms = bw.chroms()
        assert chrom in chroms, f"missing expected chromosome {chrom}"
        assert len(starts) == len(ends)
        assert np.all((0 <= starts) & (starts < ends) & (ends <= chroms[chrom]))
        assert np.all(starts[1:] >= ends[:-1])
        boundaries = np.r_[
            0, np.flatnonzero(np.diff(starts // 1_000_000)) + 1, len(starts)
        ]
        for left, right in pairwise(boundaries):
            start, end = int(starts[left]), int(ends[right - 1])
            values = np.asarray(
                bw.values(chrom, start, end, numpy=True), dtype=np.float64
            )
            assert len(values) == end - start
            assert not np.isinf(values).any()
            lo, hi = starts[left:right] - start, ends[left:right] - start
            positive = np.r_[0, np.cumsum(values >= threshold)]
            finite = np.r_[0, np.cumsum(np.isfinite(values))]
            total = np.r_[0, np.cumsum(np.nan_to_num(values))]
            conserved[left:right] = positive[hi] - positive[lo]
            coverage[left:right] = finite[hi] - finite[lo]
            means[left:right] = np.divide(
                total[hi] - total[lo],
                coverage[left:right],
                out=np.full(right - left, np.nan),
                where=coverage[left:right] > 0,
            )
    return {
        "conserved_bases": conserved,
        "label_covered_bases": coverage,
        "mean_phylop": means,
    }
