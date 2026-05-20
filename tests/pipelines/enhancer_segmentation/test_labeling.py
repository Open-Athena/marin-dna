"""Tests for per-bin labeling of segmentation windows."""

import numpy as np
import pandas as pd

from marin_dna.pipelines.enhancer_segmentation.labeling import (
    label_windows_by_bin_overlap,
)

BIN_SIZE = 128
NUM_BINS = 128
WINDOW_SIZE = BIN_SIZE * NUM_BINS  # 16384


def _single_window() -> pd.DataFrame:
    return pd.DataFrame({"chrom": ["1"], "start": [0], "end": [WINDOW_SIZE]})


def test_issue_worked_examples_bin_8_9():
    """From the issue #115 label-overlap table: each 255bp enhancer at
    these offsets should yield exactly 2 positive bins at (8, 9)."""
    windows = _single_window()
    cases = [
        (1024, 1279),  # 100% / 99% / 0%  -> bins 8, 9
        (1064, 1319),  # 69%  / 100% / 30% -> bins 8, 9
    ]
    for start, end in cases:
        positives = pd.DataFrame({"chrom": ["1"], "start": [start], "end": [end]})
        labels = label_windows_by_bin_overlap(
            windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
        )
        assert labels.shape == (1, NUM_BINS)
        positive_bins = np.flatnonzero(labels[0]).tolist()
        assert positive_bins == [8, 9], (
            f"enhancer [{start}, {end}) expected positive bins [8, 9], got {positive_bins}"
        )


def test_issue_worked_example_bin_9_10():
    """Enhancer at [1100, 1355): 41%/100%/59% overlap -> positive bins 9, 10."""
    windows = _single_window()
    positives = pd.DataFrame({"chrom": ["1"], "start": [1100], "end": [1355]})
    labels = label_windows_by_bin_overlap(
        windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
    )
    positive_bins = np.flatnonzero(labels[0]).tolist()
    assert positive_bins == [9, 10]


def test_no_positives_gives_all_zeros():
    windows = _single_window()
    positives = pd.DataFrame(columns=["chrom", "start", "end"]).astype(
        {"chrom": str, "start": int, "end": int}
    )
    labels = label_windows_by_bin_overlap(
        windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
    )
    assert labels.shape == (1, NUM_BINS)
    assert labels.sum() == 0


def test_positives_on_other_chrom_are_ignored():
    windows = _single_window()
    positives = pd.DataFrame({"chrom": ["2"], "start": [1024], "end": [1279]})
    labels = label_windows_by_bin_overlap(
        windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
    )
    assert labels.sum() == 0


def test_threshold_boundary_exact_half():
    """At exactly 50% overlap (64 bp), the bin should be labeled positive."""
    windows = _single_window()
    # Bin 0 is [0, 128). An enhancer covering [0, 64) gives exactly 64 bp
    # overlap = threshold * 128.
    positives = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [64]})
    labels = label_windows_by_bin_overlap(
        windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
    )
    assert labels[0, 0] == 1
    assert labels[0, 1:].sum() == 0


def test_threshold_boundary_below_half():
    """Just under 50% overlap (63 bp) leaves the bin negative."""
    windows = _single_window()
    positives = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [63]})
    labels = label_windows_by_bin_overlap(
        windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
    )
    assert labels.sum() == 0


def test_multiple_windows_and_positives():
    """Two windows on different chromosomes; multiple positive intervals."""
    windows = pd.DataFrame(
        {
            "chrom": ["1", "2"],
            "start": [0, 0],
            "end": [WINDOW_SIZE, WINDOW_SIZE],
        }
    )
    positives = pd.DataFrame(
        {
            "chrom": ["1", "1", "2"],
            "start": [1024, 5000, 10000],
            "end": [1279, 5100, 10300],
        }
    )
    labels = label_windows_by_bin_overlap(
        windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
    )
    assert labels.shape == (2, NUM_BINS)
    # Window 0, chrom 1: enhancer at [1024, 1279) hits bins 8, 9.
    # enhancer at [5000, 5100) is only 100bp — bin 39 [4992, 5120) gets 100bp
    # (>=64), so positive; neighboring bins are not.
    assert np.flatnonzero(labels[0]).tolist() == [8, 9, 39]
    # Window 1, chrom 2: enhancer at [10000, 10300) is 300bp — covers
    # bin 78 [9984, 10112): 112 bp, bin 79 [10112, 10240): 128 bp,
    # bin 80 [10240, 10368): 60 bp (<64 -> negative).
    assert np.flatnonzero(labels[1]).tolist() == [78, 79]


def test_multiple_windows_same_chromosome():
    """Handles multiple windows that share a chromosome."""
    windows = pd.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "start": [0, WINDOW_SIZE, 2 * WINDOW_SIZE],
            "end": [WINDOW_SIZE, 2 * WINDOW_SIZE, 3 * WINDOW_SIZE],
        }
    )
    # One enhancer in window 0 and one in window 2; window 1 has none.
    positives = pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [1024, 2 * WINDOW_SIZE + 5000],
            "end": [1279, 2 * WINDOW_SIZE + 5255],
        }
    )
    labels = label_windows_by_bin_overlap(
        windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS, threshold=0.5
    )
    assert labels.shape == (3, NUM_BINS)
    assert np.flatnonzero(labels[0]).tolist() == [8, 9]
    assert labels[1].sum() == 0
    # 2*WINDOW_SIZE + 5000 = bin 39 of window 2 (5000 // 128 = 39).
    assert np.flatnonzero(labels[2]).tolist() == [39, 40]


def test_wrong_window_size_raises():
    windows = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [WINDOW_SIZE + 1]})
    positives = pd.DataFrame(columns=["chrom", "start", "end"]).astype(
        {"chrom": str, "start": int, "end": int}
    )
    try:
        label_windows_by_bin_overlap(
            windows, positives, bin_size=BIN_SIZE, num_bins=NUM_BINS
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mis-sized window")
