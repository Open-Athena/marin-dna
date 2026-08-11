"""Tests for ``marin_dna_evals.model.interpretation`` (nucleotide dependency maps, #237).

Two flavors of model double:

- ``_PrefixSumCausalLM`` — a genuinely *causal*, content-dependent test double
  (logits at ``t`` depend only on a cumulative sum of inputs ``[0..t]``). It
  produces a rich, strictly-upper-triangular Jacobian, exercising the
  causal-invariant assert and the FWD/RC stitch with hand-checkable structure.
- The real ``hf-internal-testing/tiny-random-GPTNeoXForCausalLM`` — validates
  the actual HF ``.logits`` interface and that real causal attention zeroes the
  lower triangle.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from marin_dna_evals.model.interpretation import (
    _stitch_and_symmetrize,
    categorical_jacobian,
    dependency_matrix,
    nucleotide_dependency_map,
)
from tests.doubles import DnaTokenizerStub

TINY_CLM = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


class _PrefixSumCausalLM(nn.Module):
    """Causal test double: ``logits[:, t]`` depend only on ``input_ids[:, :t+1]``
    through a cumulative sum, so perturbing position ``i`` moves every readout
    ``m > i`` and nothing at ``m <= i``. Nonlinear (``sin``) so the four
    substitutions at a position give distinct effects. No parameters (the map
    falls back to CPU device detection)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        x = input_ids.float()
        prefix = torch.cumsum(x, dim=1)  # [B, L] — prefix[t] uses only [0..t]
        v = torch.arange(self.vocab_size, dtype=torch.float)
        logits = torch.sin(0.3 * prefix.unsqueeze(-1) + v)  # [B, L, V]
        return SimpleNamespace(logits=logits)


def _nuc_token_ids(tokenizer):
    from marin_dna.data.dna import NUCLEOTIDES
    from marin_dna_evals.transforms import _get_nucleotide_token_ids

    ids = _get_nucleotide_token_ids(tokenizer)
    return torch.tensor([ids[n] for n in NUCLEOTIDES], dtype=torch.long)


# --------------------------------------------------------------------------- #
# categorical_jacobian
# --------------------------------------------------------------------------- #
def test_categorical_jacobian_shape_and_causal_upper_triangular():
    tok = DnaTokenizerStub(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    seq = "ACGTACGT"  # W = 8
    W = len(seq)
    input_ids = torch.tensor(tok.encode(seq), dtype=torch.long)

    jac = categorical_jacobian(
        model,
        input_ids,
        nuc_token_ids=_nuc_token_ids(tok),
        n_prefix=1,
        window_size=W,
    )
    assert jac.shape == (W, 4, W, 4)

    # Causal: jac[i, :, m, :] must be exactly 0 for every target m <= i
    # (cumsum up to the readout index does not include the perturbed position).
    for i in range(W):
        for m in range(i + 1):
            assert torch.all(jac[i, :, m, :] == 0.0), f"non-causal at (i={i}, m={m})"
    # ...and genuinely non-zero somewhere above the diagonal.
    assert jac.abs().max() > 0


def test_categorical_jacobian_no_bos_blinds_first_target():
    """A no-BOS causal LM (n_prefix=0) has no prediction for the window's first
    position; its target column is zeroed and the rest stays causal upper-tri."""
    tok = DnaTokenizerStub(bos=False, eos=False)  # n_prefix == 0
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    seq = "ACGTACGT"
    W = len(seq)
    input_ids = torch.tensor(tok.encode(seq), dtype=torch.long)
    jac = categorical_jacobian(
        model,
        input_ids,
        nuc_token_ids=_nuc_token_ids(tok),
        n_prefix=0,
        window_size=W,
    )
    assert jac.shape == (W, 4, W, 4)
    # Target position 0 has no predictive distribution → its column is zeroed.
    assert torch.all(jac[:, :, 0, :] == 0.0)
    # Still causal elsewhere: jac[i,:,m,:] == 0 for m <= i; real signal above.
    for i in range(W):
        for m in range(i + 1):
            assert torch.all(jac[i, :, m, :] == 0.0), f"non-causal at (i={i}, m={m})"
    assert jac[:, :, 1:, :].abs().max() > 0


def test_categorical_jacobian_batching_invariant():
    """Result is independent of the forward batch size."""
    tok = DnaTokenizerStub(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    seq = "ACGTACGTAC"  # W = 10
    input_ids = torch.tensor(tok.encode(seq), dtype=torch.long)
    kw = {"nuc_token_ids": _nuc_token_ids(tok), "n_prefix": 1, "window_size": len(seq)}
    a = categorical_jacobian(model, input_ids, batch_size=4, **kw)
    b = categorical_jacobian(model, input_ids, batch_size=40, **kw)
    torch.testing.assert_close(a, b)


# --------------------------------------------------------------------------- #
# dependency_matrix
# --------------------------------------------------------------------------- #
def test_dependency_matrix_norm_inf_and_zero_diagonal():
    W = 3
    jac = np.zeros((W, 4, W, 4))
    jac[0, 1, 2, 3] = -5.0  # block (i=0, m=2): max-abs 5
    jac[2, 0, 0, 0] = 3.0  # block (i=2, m=0): max-abs 3
    jac[1, :, 1, :] = 9.0  # diagonal block (1,1): must be zeroed out

    D = dependency_matrix(jac, norm_ord=np.inf)
    assert D.shape == (W, W)
    assert D[0, 2] == 5.0
    assert D[2, 0] == 3.0
    assert D[1, 1] == 0.0  # diagonal explicitly zeroed
    assert np.all(np.diag(D) == 0.0)


# --------------------------------------------------------------------------- #
# _stitch_and_symmetrize
# --------------------------------------------------------------------------- #
def test_stitch_and_symmetrize_mean_and_max():
    D_fwd = np.array([[0, 2, 4], [0, 0, 6], [0, 0, 0]], dtype=float)  # upper
    D_rc = np.array([[0, 0, 0], [1, 0, 0], [3, 5, 0]], dtype=float)  # lower

    M_mean = _stitch_and_symmetrize(D_fwd, D_rc, "mean")
    assert np.allclose(M_mean, M_mean.T)
    # M[0,1] averages the two directed dependencies: (fwd 2 + rc 1)/2 = 1.5
    assert M_mean[0, 1] == 1.5
    assert M_mean[0, 2] == (4 + 3) / 2
    assert M_mean[1, 2] == (6 + 5) / 2

    M_max = _stitch_and_symmetrize(D_fwd, D_rc, "max")
    assert np.allclose(M_max, M_max.T)
    assert M_max[0, 1] == 2  # max(2, 1)
    assert M_max[0, 2] == 4  # max(4, 3)

    with pytest.raises(ValueError, match="combine"):
        _stitch_and_symmetrize(D_fwd, D_rc, "median")  # type: ignore[arg-type]


def test_stitch_mean_does_not_halve_offdiagonal_signal():
    """Regression guard for the #237 trap: a true off-diagonal pair seen by
    *one* strand keeps its magnitude (the /2 averages two real entries, it does
    not halve a single-strand signal)."""
    D_fwd = np.array([[0, 8], [0, 0]], dtype=float)  # fwd saw i=0 -> m=1 = 8
    D_rc = np.array([[0, 0], [8, 0]], dtype=float)  # rc saw i=1 -> m=0 = 8
    M = _stitch_and_symmetrize(D_fwd, D_rc, "mean")
    assert M[0, 1] == 8.0 and M[1, 0] == 8.0  # not 4.0


# --------------------------------------------------------------------------- #
# nucleotide_dependency_map (end-to-end through a model)
# --------------------------------------------------------------------------- #
def test_nucleotide_dependency_map_stitched_symmetric():
    tok = DnaTokenizerStub(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    seq = "ACGTACGT"
    W = len(seq)
    M = nucleotide_dependency_map(model, tok, seq, rc=True, combine="mean")
    assert M.shape == (W, W)
    assert np.allclose(M, M.T)
    assert np.all(np.diag(M) == 0.0)
    # The model has genuine cross-position structure → non-trivial off-diagonal.
    assert np.triu(M, 1).max() > 0


def test_nucleotide_dependency_map_rc_false_is_forward_reflection():
    tok = DnaTokenizerStub(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    seq = "ACGTACGT"
    M = nucleotide_dependency_map(model, tok, seq, rc=False)
    assert np.allclose(M, M.T)  # forward-only map, reflected
    # Differs from the stitched map (rc adds the lower triangle's real values).
    M_rc = nucleotide_dependency_map(model, tok, seq, rc=True, combine="mean")
    assert not np.allclose(M, M_rc)


def test_nucleotide_dependency_map_mean_vs_max_differ():
    tok = DnaTokenizerStub(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    seq = "ACGTACGT"
    M_mean = nucleotide_dependency_map(model, tok, seq, combine="mean")
    M_max = nucleotide_dependency_map(model, tok, seq, combine="max")
    assert not np.allclose(M_mean, M_max)
    # max >= mean elementwise (max(a,b) >= (a+b)/2 for non-negative norms).
    assert np.all(M_max + 1e-9 >= M_mean)


def test_nucleotide_dependency_map_real_tiny_clm():
    """Real HF causal LM: validates the ``.logits`` interface and that genuine
    causal attention zeroes the lower triangle (the internal asserts fire)."""
    tok = DnaTokenizerStub(bos=True, eos=True)
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM).eval()
    seq = "ACGTACGTACGT"  # W = 12
    W = len(seq)
    M = nucleotide_dependency_map(model, tok, seq, rc=True, combine="mean", atol=1e-4)
    assert M.shape == (W, W)
    assert np.allclose(M, M.T)
    assert np.all(np.diag(M) == 0.0)


def test_nucleotide_dependency_map_no_bos_is_complete():
    """No-BOS model: the forward pass can't predict window position 0 and RC
    can't predict the last — but the FWD+RC stitch recovers both, so the map is
    the full WxW size, symmetric, with no dropped (all-zero) position. This is
    what lets a no-BOS arm (e.g. exp21) stack with the BOS models at equal size."""
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    seq = "ACGTACGTAC"  # W = 10
    W = len(seq)
    tok = DnaTokenizerStub(bos=False, eos=False)  # n_prefix == 0
    M = nucleotide_dependency_map(model, tok, seq, rc=True, combine="mean")
    assert M.shape == (W, W)
    assert np.allclose(M, M.T)
    assert np.all(np.diag(M) == 0.0)
    # Every position is recovered — no all-zero row, including the two edges that
    # a single strand could not predict.
    row_sums = np.abs(M).sum(axis=1)
    assert np.all(row_sums > 0), f"dropped position(s): {np.where(row_sums == 0)[0]}"
    assert row_sums[0] > 0 and row_sums[W - 1] > 0


def test_nucleotide_dependency_map_no_bos_requires_rc():
    """A no-BOS model's blinded edge is recovered only by the RC stitch, so the
    one-sided rc=False map is rejected rather than returned silently incomplete."""
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    tok = DnaTokenizerStub(bos=False, eos=False)  # n_prefix == 0
    with pytest.raises(AssertionError, match="rc=False is unsupported"):
        nucleotide_dependency_map(model, tok, "ACGTACGT", rc=False)
