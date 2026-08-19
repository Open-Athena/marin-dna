from __future__ import annotations

from pathlib import Path

import numpy as np

from exp479_mntp.nucleotide_dependency import (
    Locus,
    mean_symmetrize,
    off_diagonal_spearman,
    plot_comparison,
)


def test_mean_symmetrization_and_off_diagonal_correlation(tmp_path: Path) -> None:
    directed = np.zeros((255, 255), dtype=np.float32)
    directed[10, 20] = 2
    directed[20, 10] = 4
    symmetric = mean_symmetrize(directed)
    assert symmetric[10, 20] == symmetric[20, 10] == 3
    assert np.diag(symmetric).sum() == 0
    assert np.isclose(off_diagonal_spearman(symmetric, symmetric * 2), 1.0)

    output = tmp_path / "comparison.svg"
    plot_comparison(
        symmetric,
        symmetric * 2,
        locus=Locus("test", "1", 100, 200, "+"),
        correlation=1.0,
        output_path=output,
    )
    assert output.read_text(encoding="utf-8").startswith("<?xml")
