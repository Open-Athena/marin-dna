"""Tests for ``compute_dependency_map`` windowing / slicing / strand handling.

The heavy compute (model load + categorical Jacobian) is mocked; these tests
pin the coordinate arithmetic — the classic off-by-one footgun — by stubbing
``nucleotide_dependency_map`` with an identifiable matrix and checking the
locus slice, genomic-coordinate index, and reverse-strand flip.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from marin_dna.pipelines.evals.interpretation import compute_dependency_map

_MOD = "marin_dna.pipelines.evals.interpretation"


@contextmanager
def _mocked_compute(window_size: int, stub_map: np.ndarray):
    """Patch loaders + the Jacobian compute so the windowing logic runs alone.

    ``Genome(...)`` returns a callable that yields a length-``window_size``
    sequence; ``nucleotide_dependency_map`` returns ``stub_map``.
    """
    genome_inst = MagicMock(return_value="A" * window_size)
    with (
        patch(f"{_MOD}.AutoTokenizer"),
        patch(f"{_MOD}.AutoModelForCausalLM"),
        patch(f"{_MOD}.Genome", return_value=genome_inst),
        patch(f"{_MOD}.torch.cuda.is_available", return_value=False),
        patch(f"{_MOD}.nucleotide_dependency_map", return_value=stub_map),
    ):
        yield genome_inst


def test_compute_dependency_map_windowing_and_coords():
    window_size = 10
    start, end = 1000, 1006  # span 6 <= 10
    # Identifiable map: M[i, j] = 100*i + j.
    stub = np.fromfunction(lambda i, j: 100 * i + j, (window_size, window_size))

    with _mocked_compute(window_size, stub) as genome_inst:
        df = compute_dependency_map(
            checkpoint_path="/ckpt",
            genome_path="/g.fa",
            chrom="19",
            start=start,
            end=end,
            strand="+",
            window_size=window_size,
        )

    # center = (1000+1006)//2 = 1003; context_start = 1003 - 5 = 998.
    genome_inst.assert_called_once_with("19", 998, 1008, strand="+")
    assert list(df.index) == list(range(start, end))
    assert list(df.columns) == list(range(start, end))
    # start_idx = 1000-998 = 2; end_idx = 1006-998 = 8.
    np.testing.assert_array_equal(df.values, stub[2:8, 2:8])


def test_compute_dependency_map_strand_minus_flips_axes():
    window_size = 10
    start, end = 1000, 1006
    stub = np.fromfunction(lambda i, j: 100 * i + j, (window_size, window_size))

    with _mocked_compute(window_size, stub):
        df = compute_dependency_map(
            checkpoint_path="/ckpt",
            genome_path="/g.fa",
            chrom="11",
            start=start,
            end=end,
            strand="-",
            window_size=window_size,
        )

    # Reverse both axes for display; index/columns run high -> low.
    np.testing.assert_array_equal(df.values, stub[2:8, 2:8][::-1, ::-1])
    assert list(df.index) == list(range(end - 1, start - 1, -1))
    assert list(df.columns) == list(range(end - 1, start - 1, -1))


def test_compute_dependency_map_rejects_oversized_locus():
    # Asserts before any model/genome load, so no mocks needed.
    with pytest.raises(AssertionError, match="exceeds model window_size"):
        compute_dependency_map(
            checkpoint_path="/ckpt",
            genome_path="/g.fa",
            chrom="1",
            start=0,
            end=20,  # span 20 > window_size 10
            strand="+",
            window_size=10,
        )
