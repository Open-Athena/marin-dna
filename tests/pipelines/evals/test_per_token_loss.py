"""Tests for the per-token loss cache (issue #296).

The forward pass needs a GPU + a checkpoint, so the HF loaders, the
special-token probe, and ``run_inference`` are mocked — this exercises the
long-format DataFrame contract: ``N * window_size`` rows, ``loss = −log p``,
``ref_base`` / ``is_upper`` reconstructed from the source case, ``pos_in_window``
tiling, ``window_id`` repeat, the special-token-layout guard, the flat-array
reshape, and the input guards.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from marin_dna.pipelines.evals.per_token_loss import compute_hf_per_token_loss


def _seqs(n: int = 2, length: int = 4) -> pd.DataFrame:
    """n identical mixed-case seqs. ``ACgt`` repeated: upper at codon-ish [0,1]."""
    base = ("ACgt" * ((length // 4) + 1))[:length]
    return pd.DataFrame(
        {"id": [f"{i}:0-{length}" for i in range(n)], "seq": [base] * n}
    )


def _patches(pred: np.ndarray) -> tuple:
    """Patch the heavy HF loaders, the special-token probe (→ 1 BOS, 0 suffix),
    and ``run_inference`` (→ ``pred``)."""
    mod = "marin_dna.pipelines.evals.per_token_loss"
    return (
        patch(f"{mod}.AutoTokenizer.from_pretrained", return_value=object()),
        patch(f"{mod}.AutoModelForCausalLM.from_pretrained", return_value=object()),
        patch(f"{mod}._get_special_token_counts", return_value=(1, 0)),
        patch(f"{mod}.run_inference", return_value=pred),
    )


def test_long_format_contract():
    n, w = 2, 4
    seqs = _seqs(n, w)  # seq "ACgt" → upper mask [T, T, F, F]
    pred = np.array(  # per-token log p, [N, W]
        [[-0.1, -0.2, -0.3, -0.4], [-1.0, -2.0, -3.0, -4.0]]
    )
    p1, p2, p3, p4 = _patches(pred)
    with p1, p2, p3, p4:
        out = compute_hf_per_token_loss("/unused", seqs, window_size=w, batch_size=2)

    assert list(out.columns) == [
        "window_id",
        "pos_in_window",
        "loss",
        "ref_base",
        "is_upper",
    ]
    assert len(out) == n * w
    # Row-major over (window, position).
    np.testing.assert_array_equal(
        out["window_id"].values, np.repeat(seqs["id"].values, w)
    )
    np.testing.assert_array_equal(out["pos_in_window"].values, np.tile(np.arange(w), n))
    # loss = −log p.
    np.testing.assert_allclose(out["loss"].values, -pred.reshape(-1))
    # ref_base uppercased; is_upper straight from source case ("ACgt").
    np.testing.assert_array_equal(out["ref_base"].values, list("ACGT") * n)
    np.testing.assert_array_equal(
        out["is_upper"].values, [True, True, False, False] * n
    )
    assert out["pos_in_window"].dtype == np.int32


def test_flat_array_reshape():
    """A flat [N*W] return (seen on some Trainer/device combos) is row-major
    reshaped back to [N, W]."""
    n, w = 2, 4
    flat = np.array([-0.1, -0.2, -0.3, -0.4, -1.0, -2.0, -3.0, -4.0])  # [N*W]
    p1, p2, p3, p4 = _patches(flat)
    with p1, p2, p3, p4:
        out = compute_hf_per_token_loss("/unused", _seqs(n, w), window_size=w)
    assert len(out) == n * w
    np.testing.assert_allclose(out["loss"].values[:4], [0.1, 0.2, 0.3, 0.4])


def test_special_token_layout_guard():
    """A non-(1 BOS, 0 suffix) layout shifts the position↔base alignment and
    must raise rather than silently mis-bin."""
    n, w = 1, 4
    pred = np.array([[-0.1, -0.2, -0.3, -0.4]])
    mod = "marin_dna.pipelines.evals.per_token_loss"
    with (
        patch(f"{mod}.AutoTokenizer.from_pretrained", return_value=object()),
        patch(f"{mod}.AutoModelForCausalLM.from_pretrained", return_value=object()),
        patch(f"{mod}._get_special_token_counts", return_value=(0, 0)),
        patch(f"{mod}.run_inference", return_value=pred),
        pytest.raises(AssertionError, match=r"255\+BOS|n_prefix"),
    ):
        compute_hf_per_token_loss("/unused", _seqs(n, w), window_size=w)


def test_window_size_mismatch_raises():
    # The seq-length guard fires before any HF load, so no mocks are needed.
    with pytest.raises(AssertionError, match="window_size"):
        compute_hf_per_token_loss("/unused", _seqs(2, 4), window_size=255)


def test_requires_id_and_seq_columns():
    with pytest.raises(AssertionError, match="id"):
        compute_hf_per_token_loss(
            "/unused", pd.DataFrame({"seq": ["ACGT"]}), window_size=4
        )
    with pytest.raises(AssertionError, match="seq"):
        compute_hf_per_token_loss(
            "/unused", pd.DataFrame({"id": ["0:0-4"]}), window_size=4
        )
