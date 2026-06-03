"""Per-position embeddings from a causal HF gLM for the ChromBPNet input layer.

This replaces ARSENAL's masked-LM ``get_avg_embeddings`` (which sees both flanks
natively) with a *causal*-LM equivalent, so a ChromBPNet-style head can be
trained on our gLM's representation instead of one-hot:

- **Tiling** — the gLM's short window (e.g. 255 bp) is tiled across ChromBPNet's
  2114 bp input, centre-out, with the ragged ends taken from the first/last
  window (mirrors ARSENAL's ``get_embeddings`` exactly).
- **Layer averaging** — within each chunk we prepend BOS, run the model with
  ``output_hidden_states=True``, and average the last ``num_layers_avg`` layers
  (ARSENAL used 6).
- **Bidirectionality** — a causal forward pass gives each position only its
  *upstream* context. We run a second pass on the **reverse complement** (each
  position's downstream context becomes upstream there), re-align it to genomic
  order, and **concatenate** → a bidirectional per-position feature of width
  ``2 * hidden``. ``bidirectional=False`` keeps the causal-only (upstream-only)
  features as an ablation.

Only the *position ordering* is reversed for the RC pass — the embedding vectors
are learned features, not one-hot, so they are not complemented. No ``no_grad``
inside, so the LM stays in the autograd graph when fine-tuning (freeze by setting
``requires_grad=False`` on the LM, exactly as the vendored ChromBPNet does).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from marin_dna.data.dna import NUCLEOTIDES, complement_base
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
)


class HFCausalChromBPNetEmbedder(nn.Module):
    """Wrap an HF causal gLM as a per-position embedder for ChromBPNet.

    ``forward(seq_ids)`` maps nucleotide **token ids** ``[B, seq_input_size]``
    (in the model's vocab, no BOS) to per-position embeddings
    ``[B, seq_input_size, out_dim]`` where ``out_dim = hidden * (2 if
    bidirectional else 1)``.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        seq_input_size: int = 2114,
        chunk_size: int = 255,
        num_layers_avg: int = 6,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        assert 0 < chunk_size <= seq_input_size, (
            f"chunk_size {chunk_size} must be in (0, seq_input_size={seq_input_size}]"
        )
        self.model = model
        self.seq_input_size = seq_input_size
        self.chunk_size = chunk_size
        self.num_layers_avg = num_layers_avg
        self.bidirectional = bidirectional

        n_prefix, n_suffix = _get_special_token_counts(tokenizer)
        self.n_prefix = n_prefix
        self.n_suffix = n_suffix
        # Special-token ids to wrap each chunk, matching the tokenizer's own
        # policy (probe on a single base: prefix + [base] + suffix).
        probe = tokenizer.encode("A")
        self._prefix_ids: list[int] = probe[:n_prefix]
        self._suffix_ids: list[int] = probe[n_prefix + 1 :] if n_suffix else []

        nuc_to_id = _get_nucleotide_token_ids(tokenizer)  # {A,C,G,T -> id}
        cfg: Any = model.config
        self.hidden_size = int(cfg.hidden_size)
        self.out_dim = self.hidden_size * (2 if bidirectional else 1)

        # Complement lookup over the whole vocab: id(A)<->id(T), id(C)<->id(G);
        # every other id (N, specials) maps to itself. Registered as a buffer so
        # it follows .to(device).
        vocab_size = int(getattr(cfg, "vocab_size", max(nuc_to_id.values()) + 1))
        comp = torch.arange(vocab_size, dtype=torch.long)
        for nuc in NUCLEOTIDES:
            comp[nuc_to_id[nuc]] = nuc_to_id[complement_base(nuc)]
        self.register_buffer("_complement_lut", comp, persistent=False)

    # -- one chunk -----------------------------------------------------------
    def _embed_chunk(self, chunk_ids: torch.Tensor) -> torch.Tensor:
        """``[B, chunk_len]`` nucleotide ids -> ``[B, chunk_len, hidden]``.

        Prepends/append the tokenizer's special tokens, runs one forward pass,
        averages the last ``num_layers_avg`` hidden-state layers, and strips the
        special-token positions back off.
        """
        b, chunk_len = chunk_ids.shape
        device = chunk_ids.device
        pieces = [chunk_ids]
        if self._prefix_ids:
            pre = torch.tensor(self._prefix_ids, device=device).expand(b, -1)
            pieces.insert(0, pre)
        if self._suffix_ids:
            suf = torch.tensor(self._suffix_ids, device=device).expand(b, -1)
            pieces.append(suf)
        input_ids = torch.cat(pieces, dim=1)

        hidden_states = self.model(input_ids, output_hidden_states=True).hidden_states
        # hidden_states: tuple len (n_layers + 1); take the last num_layers_avg.
        n = min(self.num_layers_avg, len(hidden_states))
        avg = torch.stack(hidden_states[-n:], dim=0).mean(dim=0)  # [B, T, hidden]
        return avg[:, self.n_prefix : self.n_prefix + chunk_len, :]

    # -- one strand (centre-out tiling, mirrors ARSENAL get_embeddings) ------
    def _embed_strand(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """``[B, seq_input_size]`` -> ``[B, seq_input_size, hidden]``."""
        seq_len, chunk = self.seq_input_size, self.chunk_size
        assert seq_ids.shape[1] == seq_len, (
            f"expected seq length {seq_len}, got {seq_ids.shape[1]}"
        )
        if seq_len == chunk:
            return self._embed_chunk(seq_ids)

        center = seq_len // 2
        first_start = center - chunk // 2
        first_end = first_start + chunk
        embs = self._embed_chunk(seq_ids[:, first_start:first_end])

        # Expand right.
        right_start = first_end
        while right_start < seq_len:
            if right_start + chunk <= seq_len:
                embs = torch.cat(
                    [
                        embs,
                        self._embed_chunk(
                            seq_ids[:, right_start : right_start + chunk]
                        ),
                    ],
                    dim=1,
                )
                right_start += chunk
            else:
                remainder = seq_len - right_start
                end_embs = self._embed_chunk(seq_ids[:, -chunk:])
                embs = torch.cat([embs, end_embs[:, -remainder:]], dim=1)
                break

        # Expand left (prepend).
        left_end = first_start
        left_chunks: list[torch.Tensor] = []
        while left_end > 0:
            if left_end - chunk >= 0:
                left_chunks.insert(
                    0, self._embed_chunk(seq_ids[:, left_end - chunk : left_end])
                )
                left_end -= chunk
            else:
                remainder = left_end
                start_embs = self._embed_chunk(seq_ids[:, :chunk])
                left_chunks.insert(0, start_embs[:, :remainder])
                break
        if left_chunks:
            embs = torch.cat(left_chunks + [embs], dim=1)
        return embs

    def reverse_complement_ids(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """RC in token space: reverse the position order and complement bases."""
        lut = cast(torch.Tensor, self._complement_lut)
        return lut[torch.flip(seq_ids, dims=[1])]

    def forward(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """``[B, seq_input_size]`` nucleotide ids -> ``[B, seq_input_size, out_dim]``."""
        fwd = self._embed_strand(seq_ids)
        if not self.bidirectional:
            return fwd
        rc = self._embed_strand(self.reverse_complement_ids(seq_ids))
        rc_aligned = torch.flip(rc, dims=[1])  # back to genomic position order
        return torch.cat([fwd, rc_aligned], dim=-1)
