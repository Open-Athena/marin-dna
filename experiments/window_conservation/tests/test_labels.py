from pathlib import Path

import numpy as np
import pyBigWig

from window_conservation.labels import summarize, window_labels


def test_threshold_and_missing_policy() -> None:
    assert summarize(np.array([2.0, np.nan, -1.0, 3.0]), 2.0)[:2] == (2, 3)
    assert summarize(np.array([np.nan]), -1.0)[:2] == (0, 0)


def test_bigwig_zero_based_half_open(tmp_path: Path) -> None:
    path = tmp_path / "labels.bw"
    with pyBigWig.open(str(path), "w") as bw:
        bw.addHeader([("chr1", 20)])
        bw.addEntries(["chr1", "chr1"], [0, 10], ends=[5, 20], values=[2.0, -1.0])
    labels = window_labels(
        path, "chr1", np.array([0, 5, 10]), np.array([5, 10, 20]), 2.0
    )
    assert labels["conserved_bases"].tolist() == [5, 0, 0]
    assert labels["label_covered_bases"].tolist() == [5, 0, 10]


def test_bigwig_batching_preserves_gaps_and_chunk_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "gapped.bw"
    with pyBigWig.open(str(path), "w") as bw:
        bw.addHeader([("chr1", 2_000_100)])
        bw.addEntries(
            ["chr1"] * 3,
            [999_950, 1_000_075, 2_000_000],
            ends=[1_000_025, 1_000_100, 2_000_100],
            values=[3.0, -1.0, 4.0],
        )
    labels = window_labels(
        path,
        "chr1",
        np.array([999_900, 1_000_000, 2_000_000]),
        np.array([1_000_000, 1_000_100, 2_000_100]),
        3.0,
    )
    assert labels["conserved_bases"].tolist() == [50, 25, 100]
    assert labels["label_covered_bases"].tolist() == [50, 50, 100]
    np.testing.assert_allclose(labels["mean_phylop"], [3.0, 1.0, 4.0])
