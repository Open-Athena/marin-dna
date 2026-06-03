"""Nucleotide dependency maps via the categorical Jacobian (issue #237).

The categorical Jacobian measures how a single-nucleotide substitution at
position ``i`` perturbs the model's predicted nucleotide distribution at every
other position ``j``. Collapsing each ``4x4`` block to a scalar gives an
``L x L`` *dependency map* (Tomaz da Silva et al., "Nucleotide dependency
analysis of genomic language models detects functional elements",
*Nat. Genet.* 2025), ported here from GPN-Star's ``interpretation/`` analysis.

**Autoregressive twist.** GPN-Star is a masked LM, so its Jacobian is naturally
two-sided. Our models are *causal* (``AutoModelForCausalLM``): the prediction
for position ``t`` is ``P(x_t | x_{<t})`` and depends only on the left context.
So the forward-strand Jacobian is **strictly upper-triangular** (perturbing
``i`` cannot move any target ``m <= i``; the diagonal is structurally ~0). To
recover the lower triangle we run the reverse-complemented window, flip it back
to forward coordinates (now strictly lower-triangular), **stitch** the two
disjoint triangles into one matrix ``A``, and **symmetrize** — ``(A + A.T)/2``
(``combine="mean"``, the default) or ``max(A, A.T)`` (``combine="max"``, the
paper's "max dependency within a pair"). See issue #237 for the full design.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from marin_dna.data.dna import NUCLEOTIDES, reverse_complement
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
)


@torch.no_grad()
def categorical_jacobian(
    model: Any,
    input_ids: Int[Tensor, " L"],
    *,
    nuc_token_ids: Int[Tensor, " 4"],
    n_prefix: int,
    window_size: int,
    batch_size: int = 32,
) -> Float[Tensor, "W 4 W 4"]:
    """Categorical Jacobian of a causal LM over a window's DNA positions.

    ``jac[i, a, m, b]`` is
    ``log p(base b at DNA pos m | DNA pos i := nuc a)
      - log p(base b at DNA pos m | wild-type)``,
    computed in 4-nucleotide log-softmax space. For a causal LM ``jac`` is
    strictly upper-triangular in ``(i, m)``.

    Args:
        model: HF-shaped causal LM — ``model(input_ids).logits`` must return
            ``[B, L, V]`` (duck-typed; ``AutoModelForCausalLM`` satisfies it).
        input_ids: One tokenized window, shape ``[L]`` (includes BOS / any
            special tokens). Must already live on the model's device.
        nuc_token_ids: Length-4 tensor of A/C/G/T token IDs in ``NUCLEOTIDES``
            order.
        n_prefix: Number of auto-prepended special tokens (BOS). Must be ``>= 1``
            so DNA position 0 has a predictive context.
        window_size: Number of DNA bases ``W`` in the window.
        batch_size: Sequences per forward pass.

    Returns:
        ``[W, 4, W, 4]`` Jacobian on ``input_ids.device`` (fp32).
    """
    device = input_ids.device
    W = window_size
    L = int(input_ids.shape[0])
    assert n_prefix >= 1, (
        "categorical_jacobian needs a BOS token (n_prefix>=1): under a causal LM "
        "with no left context, DNA position 0 has no predictive distribution. "
        "exp135-style char tokenizers prepend BOS; a no-BOS model is unsupported."
    )
    assert L >= n_prefix + W, (
        f"input_ids length {L} too short for window {W} + prefix {n_prefix}"
    )
    nuc_token_ids = nuc_token_ids.to(device)
    # The logit that predicts DNA position m sits at index (n_prefix + m - 1):
    # next-token logits[:, k] predict the token at input index k+1, and DNA
    # position m lives at input index n_prefix + m.
    readout_idx = torch.arange(W, device=device) + (n_prefix - 1)

    def _readout(batch_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B W 4"]:
        logits = model(batch_ids).logits  # [B, L, V]
        sel = logits[:, readout_idx][..., nuc_token_ids]  # [B, W, 4]
        # fp32 log_softmax (biofoundation #21): bf16 rounding would otherwise
        # leak into the near-zero lower triangle and break the causal assert.
        return F.log_softmax(sel.float(), dim=-1)

    base = _readout(input_ids.unsqueeze(0))[0]  # [W, 4]

    # All single-position substitutions: pert[i, a] = input_ids with DNA pos i
    # set to nucleotide a. Shape [W, 4, L].
    pert = input_ids.view(1, 1, L).repeat(W, 4, 1)
    for i in range(W):
        pert[i, :, n_prefix + i] = nuc_token_ids
    pert = pert.reshape(W * 4, L)

    out = torch.empty((W * 4, W, 4), dtype=torch.float32, device=device)
    for s in range(0, W * 4, batch_size):
        out[s : s + batch_size] = _readout(pert[s : s + batch_size])

    return out.reshape(W, 4, W, 4) - base.view(1, 1, W, 4)


def dependency_matrix(jac: np.ndarray, norm_ord: float = np.inf) -> np.ndarray:
    """Collapse each ``4x4`` block of a ``[W, 4, W, 4]`` Jacobian to a scalar
    dependency via a vector norm over the 16 entries; zero the diagonal.

    Returns a ``[W, W]`` matrix. For a forward (causal) Jacobian the result is
    upper-triangular; the diagonal is zeroed explicitly (it is structurally ~0
    for a causal LM but we zero it to match the masked-LM reference).
    """
    W = jac.shape[0]
    assert jac.shape == (W, 4, W, 4), f"expected [W,4,W,4], got {jac.shape}"
    x = jac.transpose(0, 2, 1, 3).reshape(W, W, -1)  # [W, W, 16]
    D = np.linalg.norm(x, ord=norm_ord, axis=2)
    np.fill_diagonal(D, 0.0)
    return D


def _stitch_and_symmetrize(
    D_fwd: np.ndarray,
    D_rc_flipped: np.ndarray,
    combine: Literal["mean", "max"],
) -> np.ndarray:
    """Stitch the forward (upper-triangular) and reverse-complement
    (lower-triangular, already flipped to forward coords) dependency matrices
    into one matrix, then symmetrize.

    The two triangles have disjoint support, so an elementwise **sum** stitches
    them (this is *not* a per-strand average — that would halve every
    off-diagonal value). The ``/2`` in ``mean`` is the symmetrization of the
    stitched matrix: it averages the two *directed* dependencies ``i->j`` (from
    the forward strand) and ``j->i`` (from RC) of each pair. See issue #237.
    """
    A = D_fwd + D_rc_flipped
    if combine == "mean":
        return (A + A.T) / 2.0
    if combine == "max":
        return np.maximum(A, A.T)
    raise ValueError(f"combine must be 'mean' or 'max', got {combine!r}")


@torch.no_grad()
def nucleotide_dependency_map(
    model: Any,
    tokenizer: Any,
    seq: str,
    *,
    rc: bool = True,
    combine: Literal["mean", "max"] = "mean",
    norm_ord: float = np.inf,
    batch_size: int = 32,
    atol: float = 1e-3,
) -> np.ndarray:
    """Nucleotide dependency map for one window, FWD+RC-stitched for causal LMs.

    Args:
        model: HF-shaped causal LM (see ``categorical_jacobian``).
        tokenizer: Tokenizer for ``model`` (encodes ``seq``; supplies BOS count
            and A/C/G/T token IDs).
        seq: DNA window of length ``W``. The model receives the full window as
            context.
        rc: If True (default), also run the reverse-complemented window and
            stitch FWD (upper) + RC (lower) before symmetrizing. If False,
            return the forward-only map reflected via ``max`` — a one-sided
            diagnostic, not a true dependency map (a causal LM sees only the
            upper triangle on a single strand).
        combine: Symmetrization of the stitched matrix when ``rc=True`` —
            ``"mean"`` = ``(A + A.T)/2`` (default), ``"max"`` = ``max(A, A.T)``.
        norm_ord: Vector-norm order collapsing each ``4x4`` block (``np.inf`` =
            max-abs, as in GPN-Star).
        batch_size: Sequences per forward pass.
        atol: Tolerance for the causal-triangularity asserts.

    Returns:
        Symmetric ``[W, W]`` dependency map (numpy fp32/fp64).
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    n_prefix, _ = _get_special_token_counts(tokenizer)
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    nuc_token_ids = torch.tensor(
        [nuc_ids[nuc] for nuc in NUCLEOTIDES], dtype=torch.long
    )
    W = len(seq)

    def _strand_dependency(s: str) -> np.ndarray:
        ids = torch.tensor(tokenizer.encode(s), dtype=torch.long, device=device)
        jac = categorical_jacobian(
            model,
            ids,
            nuc_token_ids=nuc_token_ids,
            n_prefix=n_prefix,
            window_size=W,
            batch_size=batch_size,
        )
        return dependency_matrix(jac.cpu().numpy(), norm_ord=norm_ord)

    D_fwd = _strand_dependency(seq)
    # Causal invariant: the forward Jacobian is strictly upper-triangular.
    # np.tril includes the (already-zeroed) diagonal; the strictly-lower part
    # must be ~0 or our token/strand indexing is wrong.
    lower_fwd = np.tril(D_fwd)
    assert np.allclose(lower_fwd, 0.0, atol=atol), (
        f"forward dependency map not upper-triangular (max |lower-tri| = "
        f"{np.abs(lower_fwd).max():.2e} > atol={atol}); causal readout indexing "
        f"is likely wrong"
    )

    if not rc:
        return np.maximum(D_fwd, D_fwd.T)

    D_rc = _strand_dependency(reverse_complement(seq))
    # RC position k corresponds to forward position W-1-k; flip both axes to
    # bring the RC map into forward coordinates. It is now lower-triangular.
    D_rc_flipped = D_rc[::-1, ::-1]
    upper_rc = np.triu(D_rc_flipped)
    assert np.allclose(upper_rc, 0.0, atol=atol), (
        f"reverse-complement dependency map not lower-triangular after flip "
        f"(max |upper-tri| = {np.abs(upper_rc).max():.2e} > atol={atol}); the RC "
        f"coordinate mapping is likely wrong"
    )

    M = _stitch_and_symmetrize(D_fwd, D_rc_flipped, combine)
    assert np.allclose(M, M.T, atol=1e-6), "dependency map is not symmetric"
    return M
