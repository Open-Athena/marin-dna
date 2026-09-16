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
    labels = window_labels(path, "chr1", np.array([0, 5, 10]), np.array([5, 10, 20]), 2.0)
    assert labels["conserved_bases"].tolist() == [5, 0, 0]
    assert labels["label_covered_bases"].tolist() == [5, 0, 10]
