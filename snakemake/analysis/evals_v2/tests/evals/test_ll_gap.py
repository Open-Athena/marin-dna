"""Tests for the HF LL-gap eval (issue #274).

The real forward pass needs a GPU + a downloaded checkpoint, so the model,
tokenizer, and compute kernel are mocked — this exercises the DataFrame
contract, the ``id`` passthrough, the ``window_size`` guard, the missing-column
guard, and the flat-array reshape. The pure-numpy aggregator (``aggregate_ll_gap``)
is checked separately by ``test_aggregate_ll_gap_canonical_import_and_gap_sign`` below.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.ll_gap import aggregate_ll_gap, compute_hf_ll_gap


def _seqs(n: int = 3, length: int = 8) -> pd.DataFrame:
    """n identical mixed-case seqs of the given length (``ACGTacgt`` repeated)."""
    base = ("ACGTacgt" * ((length // 8) + 1))[:length]
    return pd.DataFrame({"id": [f"r{i}" for i in range(n)], "seq": [base] * n})


def _patched_load():
    """Patch the heavy HF loaders so no checkpoint is ever downloaded."""
    return patch(
        "marin_dna_evals.ll_gap.load_hf_causal_lm_and_tokenizer",
        return_value=(object(), object()),
    )


def test_compute_hf_ll_gap_returns_atoms_with_id():
    seqs = _seqs(n=3, length=8)
    pred = np.array(
        [[-2.0, -5.0, 4, 3], [-1.0, -4.0, 4, 3], [-3.0, -6.0, 4, 3]], dtype=float
    )
    load = _patched_load()
    with (
        load,
        patch("marin_dna_evals.ll_gap.run_ll_clm", return_value=pred),
    ):
        out = compute_hf_ll_gap("/unused", seqs, window_size=8, batch_size=2)

    assert list(out.columns) == [
        "id",
        "ll_sum_upper",
        "ll_sum_lower",
        "n_upper",
        "n_lower",
    ]
    assert len(out) == 3
    np.testing.assert_array_equal(out["id"].values, seqs["id"].values)
    np.testing.assert_array_equal(out["ll_sum_upper"].values, pred[:, 0])
    np.testing.assert_array_equal(out["n_upper"].values, pred[:, 2].astype(int))
    # Sums kept as fp64, counts as int64.
    assert out["ll_sum_upper"].dtype == np.float64
    assert out["n_upper"].dtype == np.int64


def test_compute_hf_ll_gap_threads_compile_and_bf16():
    seqs = _seqs(n=2, length=8)
    pred = np.zeros((2, 4), dtype=float)
    load = _patched_load()
    with (
        load,
        patch("marin_dna_evals.ll_gap.run_ll_clm", return_value=pred) as runner,
    ):
        compute_hf_ll_gap(
            "/unused",
            seqs,
            window_size=8,
            batch_size=2,
            torch_compile=True,
            bf16=False,
        )

    kwargs = runner.call_args.kwargs["inference_kwargs"]
    assert kwargs["torch_compile"] is True
    assert kwargs["bf16_full_eval"] is False


def test_compute_hf_ll_gap_reshapes_flat_array():
    """A flat ``[N*4]`` return (seen on some Trainer/device combos) is row-major
    reshaped back to ``[N, 4]`` preserving row order."""
    seqs = _seqs(n=2, length=8)
    flat = np.array([-1.0, -2.0, 4, 3, -1.5, -2.5, 4, 3])  # [N*4]
    load = _patched_load()
    with (
        load,
        patch("marin_dna_evals.ll_gap.run_ll_clm", return_value=flat),
    ):
        out = compute_hf_ll_gap("/unused", seqs, window_size=8, batch_size=2)

    assert len(out) == 2
    np.testing.assert_array_equal(out["ll_sum_upper"].values, [-1.0, -1.5])
    np.testing.assert_array_equal(out["ll_sum_lower"].values, [-2.0, -2.5])


def test_compute_hf_ll_gap_window_size_mismatch_raises():
    seqs = _seqs(n=2, length=8)
    with pytest.raises(AssertionError, match="window_size"):
        compute_hf_ll_gap("/unused", seqs, window_size=255, batch_size=2)


def test_compute_hf_ll_gap_requires_seq_column():
    seqs = pd.DataFrame({"id": ["a"], "sequence": ["ACGTacgt"]})
    with pytest.raises(AssertionError, match="seq"):
        compute_hf_ll_gap("/unused", seqs, window_size=8)


def test_aggregate_ll_gap_canonical_import_and_gap_sign():
    """aggregate_ll_gap is importable from its model-agnostic home and the gap
    convention is ``LL_upper - LL_lower`` (positive ⇒ functional easier)."""
    pred = np.array([[-2.0, 0.0, 4.0, 0.0], [0.0, -9.0, 0.0, 6.0]], dtype=np.float32)
    out = aggregate_ll_gap(pred)
    assert out["LL_upper"] == -0.5
    assert out["LL_lower"] == -1.5
    assert out["gap"] == 1.0
    assert out["n_upper"] == 4 and out["n_lower"] == 6
