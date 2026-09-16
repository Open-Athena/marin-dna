"""Reuse the projection pipeline's phyloP >= 2.2162 whole-window definition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBigWig


def summarize(values: np.ndarray, threshold: float) -> tuple[int, int, float]:
    finite = np.isfinite(values)
    assert not np.isinf(values).any()
    return int((values >= threshold).sum()), int(finite.sum()), float(np.mean(values[finite])) if finite.any() else float("nan")


def window_labels(path: Path, chrom: str, starts: np.ndarray, ends: np.ndarray,
                  threshold: float) -> dict[str, np.ndarray]:
    conserved, coverage, means = [], [], []
    with pyBigWig.open(str(path)) as bw:
        chroms = bw.chroms()
        assert chrom in chroms, f"missing expected chromosome {chrom}"
        for start, end in zip(starts, ends, strict=True):
            assert 0 <= start < end <= chroms[chrom]
            values = np.asarray(bw.values(chrom, int(start), int(end), numpy=True), dtype=np.float64)
            assert len(values) == end - start
            positive, valid, mean = summarize(values, threshold)
            conserved.append(positive)
            coverage.append(valid)
            means.append(mean)
    return {"conserved_bases": np.array(conserved, dtype=np.int32),
            "label_covered_bases": np.array(coverage, dtype=np.int32), "mean_phylop": np.array(means)}
