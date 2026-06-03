"""Window embeddings for genomic sequences (issue #246).

Extract one fixed-length embedding per DNA window from a gLM: expand the window
to the model's context, take per-position hidden states from one layer,
mean-pool the center ``n_center_bp`` positions, and average the forward and
reverse-complement strands. Ported from GPN-Star's embedding-UMAP analysis (Ye
et al., bioRxiv 2025; their Fig 4A/4B) to our autoregressive
``AutoModelForCausalLM`` checkpoints.

**Autoregressive note.** A causal LM's hidden state at position ``p`` encodes
only ``x_{<=p}`` (left context), so a single-strand center mean-pool is
left-context-biased. Averaging the forward and reverse-complement embeddings
symmetrizes this — each strand supplies the opposite-direction context.
(GPN-Star is bidirectional and averages strands for invariance; for us it
additionally corrects the causal bias.) Unlike the nucleotide-dependency map
(``model.interpretation``), no coordinate flip is needed: we mean-pool a
*symmetric* central block, so position order within the pool is irrelevant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from marin_dna.data.dna import reverse_complement
from marin_dna.data.transforms import _get_special_token_counts


@torch.no_grad()
def window_embeddings(
    model: Any,
    tokenizer: Any,
    seqs: list[str],
    *,
    layer_index: int = -1,
    n_center_bp: int = 100,
    rc: bool = True,
    batch_size: int = 64,
) -> np.ndarray:
    """Center-pooled, FWD+RC-averaged embeddings for a batch of equal-length windows.

    Each sequence in ``seqs`` is one DNA window of identical length ``W``
    (already expanded to the model's context). The returned row for window
    ``k`` is the mean over its center ``n_center_bp`` positions of one layer's
    hidden states, averaged across the forward and reverse-complement strands.

    Args:
        model: HF-shaped model. ``model(input_ids, output_hidden_states=True)``
            must return ``.hidden_states`` — a tuple of ``[B, L, D]`` tensors
            ``(embeddings, layer_1, …, layer_N)``. ``AutoModelForCausalLM`` and
            base ``AutoModel`` both satisfy it. Must be in eval mode on-device.
        tokenizer: Tokenizer for ``model`` (encodes each window; supplies the
            BOS/special-token prefix count via ``_get_special_token_counts``).
        seqs: Windows of equal length ``W`` (uppercase ACGT; no ``N`` — the
            caller filters those). Requires ``W >= n_center_bp``.
        layer_index: Which ``hidden_states`` layer to read (``-1`` = last layer,
            matching GPN-Star).
        n_center_bp: Number of central DNA positions to mean-pool.
        rc: If True (default), also embed the reverse complement of each window
            and average the two ``[D]`` vectors.
        batch_size: Sequences per forward pass (each strand batched separately).

    Returns:
        ``[N, D]`` float32 numpy array (``N = len(seqs)``).
    """
    assert seqs, "seqs must be non-empty"
    W = len(seqs[0])
    assert all(len(s) == W for s in seqs), "all windows must have equal length"
    assert n_center_bp <= W, f"n_center_bp {n_center_bp} exceeds window length {W}"
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    n_prefix, _ = _get_special_token_counts(tokenizer)

    # Center DNA positions ``[c0, c0 + n_center_bp)`` in window coordinates,
    # offset by ``n_prefix`` into token coordinates. The center block is
    # symmetric about the window midpoint, so the same slice indexes the
    # corresponding bases on the reverse-complement strand.
    c0 = (W - n_center_bp) // 2
    tok_lo = n_prefix + c0
    tok_hi = tok_lo + n_center_bp

    def _pool(batch_seqs: list[str]) -> Tensor:
        ids = torch.tensor(
            [tokenizer.encode(s) for s in batch_seqs],
            dtype=torch.long,
            device=device,
        )
        hs = model(ids, output_hidden_states=True).hidden_states[layer_index]
        # fp32 mean: bf16 accumulation over the pooled positions would add
        # ~1e-3 noise (cf. the log_softmax fp32 cast in model/scoring.py).
        return hs[:, tok_lo:tok_hi].float().mean(dim=1)  # [B, D]

    strands = [seqs]
    if rc:
        strands.append([reverse_complement(s) for s in seqs])

    acc: Tensor | None = None
    for strand_seqs in strands:
        pooled: list[Tensor] = []
        for s in range(0, len(strand_seqs), batch_size):
            pooled.append(_pool(strand_seqs[s : s + batch_size]).cpu())
        emb = torch.cat(pooled, dim=0)  # [N, D] — row k aligns across strands
        acc = emb if acc is None else acc + emb
    assert acc is not None
    out = (acc / len(strands)).numpy().astype(np.float32)
    assert out.shape[0] == len(seqs), (
        f"got {out.shape[0]} embedding rows, expected {len(seqs)}"
    )
    assert np.isfinite(out).all(), "embeddings contain non-finite values"
    return out
