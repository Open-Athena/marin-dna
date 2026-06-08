"""Unit tests for the per-variant ref/alt embedding extraction (#302)."""

from __future__ import annotations

import torch
import torch.nn as nn

from marin_dna.data.dna import reverse_complement
from marin_dna.pipelines.evals.variant_embedding import (
    compute_variant_embeddings,
    resolve_layer_indices,
)

_VOCAB = {"[BOS]": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}


class _FakeTokenizer:
    bos_token_id = 0
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id] + [_VOCAB[c] for c in text]


class _FakeGenome:
    """Returns substrings of a per-chromosome reference; RC on the minus strand."""

    def __init__(self, seqs: dict[str, str]):
        self.seqs = seqs

    def __call__(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        sub = self.seqs[str(chrom)][start:end]
        return reverse_complement(sub) if strand == "-" else sub


class _FakeModel(nn.Module):
    """hidden_states[k][b, l, :] = input_id[b, l] + k  (so layers & tokens are recoverable)."""

    def __init__(self, num_hidden_layers: int = 3, hidden: int = 8):
        super().__init__()
        self.config = type("cfg", (), {"num_hidden_layers": num_hidden_layers})()
        self._h = hidden
        self._n = num_hidden_layers
        self.register_parameter(
            "_p", nn.Parameter(torch.zeros(1))
        )  # so .to(dtype) works

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        base = input_ids[:, :, None].float().expand(-1, -1, self._h)
        hs = tuple(base + k for k in range(self._n + 1))
        return type("out", (), {"hidden_states": hs})()


def test_resolve_layer_indices():
    assert resolve_layer_indices(12, (0.25, 0.5, 0.75, 1.0)) == [3, 6, 9, 12]
    # clamp + dedup on a tiny network
    assert resolve_layer_indices(2, (0.25, 0.5, 0.75, 1.0)) == [1, 2]


def test_ref_alt_pool_and_swap():
    # window 7 (odd, symmetric): variant at in-seq index 3 → token index 4 (after BOS).
    # center 3 pool → token slice [3:6] (indices 3,4,5), which includes the variant token.
    ws, ncenter = 7, 3
    chrom = "1"
    ref_seq = "ACGTACGTACGTACGT"
    genome = _FakeGenome({chrom: ref_seq})
    # pos is 1-based; pick pos=8 → center_index=7 → window ref_seq[4:11]="ACGTACG", center base ref_seq[7]='T'
    variants = [{"chrom": chrom, "pos": 8, "ref": ref_seq[7], "alt": "G"}]
    assert ref_seq[7] == "T"

    model, tok = _FakeModel(num_hidden_layers=3), _FakeTokenizer()
    ref, alt = compute_variant_embeddings(
        model,
        tok,
        genome,
        variants,
        ws,
        layer_indices=[3],
        n_center_bp=ncenter,
        rc=False,
        batch_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    assert ref.shape == (1, 1, 8) and alt.shape == (1, 1, 8)

    # tokens of the ref window "ACGTACG" = [1,2,3,4,1,2,3]; with BOS → [0,1,2,3,4,1,2,3]
    ids = [0, 1, 2, 3, 4, 1, 2, 3]
    # pooled center = tokens at indices [3,4,5] = (3,4,1); last layer adds k=3
    expected_ref = (3 + 4 + 1) / 3 + 3
    assert abs(ref[0, 0, 0] - expected_ref) < 1e-5, (ref[0, 0, 0], expected_ref)
    # alt swaps token at index 4 (the variant) from 4 ('T') to G=3; only that position changes
    expected_alt = (3 + 3 + 1) / 3 + 3
    assert abs(alt[0, 0, 0] - expected_alt) < 1e-5, (alt[0, 0, 0], expected_alt)
    # delta is concentrated and nonzero exactly by (alt_tok - ref_tok)/n_center
    assert abs((alt[0, 0, 0] - ref[0, 0, 0]) - (3 - 4) / 3) < 1e-5
    _ = ids


def test_fwd_rc_average_runs():
    ws = 7
    chrom = "1"
    ref_seq = "ACGTACGTACGTACGT"
    genome = _FakeGenome({chrom: ref_seq})
    variants = [{"chrom": chrom, "pos": 8, "ref": ref_seq[7], "alt": "A"}]
    model, tok = _FakeModel(), _FakeTokenizer()
    ref, alt = compute_variant_embeddings(
        model,
        tok,
        genome,
        variants,
        ws,
        layer_indices=[2, 3],
        n_center_bp=3,
        rc=True,
        batch_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    assert ref.shape == (1, 2, 8) and alt.shape == (1, 2, 8)
