"""Tests for the window-embedding harness pieces (#246):

- ``compute_window_embedding`` (model/scoring.py): center mean-pool + layer select.
- ``transform_window_embedding`` (data/transforms.py): centered window + strand.

The full ``run_window_embeddings`` path (HF Trainer) is exercised by the
pipeline smoke run; these are the fast CPU unit tests of the two new kernels.
"""

from types import SimpleNamespace

import torch
from torch import nn

from marin_dna.data.transforms import transform_window_embedding
from marin_dna.model.runner import _center_token_bounds
from marin_dna.model.scoring import compute_window_embedding
from marin_dna.tokenizer.char import create_char_tokenizer


class _HiddenStateModel(nn.Module):
    """Hidden state at ``(layer L, position t)`` = ``[token_id_at_t, L]``.
    ``last_hidden_state`` is the final layer; ``output_hidden_states`` returns the
    full tuple. Lets us hand-check center pooling (channel 0) and layer selection
    (channel 1)."""

    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers

    def _layer(self, tok: torch.Tensor, idx: int) -> torch.Tensor:
        return torch.stack([tok, torch.full_like(tok, float(idx))], dim=-1)

    def forward(self, input_ids, output_hidden_states: bool = False):
        tok = input_ids.float()  # [B, L]
        last = self._layer(tok, self.n_layers)
        if output_hidden_states:
            hs = tuple(self._layer(tok, i) for i in range(self.n_layers + 1))
            return SimpleNamespace(last_hidden_state=last, hidden_states=hs)
        return SimpleNamespace(last_hidden_state=last)


def test_compute_window_embedding_center_pool_last_layer():
    model = _HiddenStateModel(n_layers=3)
    ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]])
    emb = compute_window_embedding(model, ids, tok_lo=2, tok_hi=5, layer_index=-1)
    assert emb.shape == (2, 2)
    # channel 0 = mean token id over the pooled center slice [2, 5)
    assert torch.allclose(emb[:, 0], ids[:, 2:5].float().mean(1))
    # channel 1 = last-layer index (n_layers = 3)
    assert torch.allclose(emb[:, 1], torch.tensor([3.0, 3.0]))


def test_compute_window_embedding_layer_index_selects_layer():
    model = _HiddenStateModel(n_layers=3)
    ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]])
    for layer_index, expected in [(0, 0.0), (2, 2.0), (-1, 3.0)]:
        emb = compute_window_embedding(
            model, ids, tok_lo=1, tok_hi=4, layer_index=layer_index
        )
        assert torch.allclose(emb[:, 1], torch.tensor([expected]))


class _RecordingGenome:
    """Fake Genome: records calls; returns a fixed-length strand-coded sequence."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        self.calls.append((chrom, start, end, strand))
        return ("A" if strand == "+" else "C") * (end - start)


def test_transform_window_embedding_centers_and_tokenizes():
    tok = create_char_tokenizer(bos=True, eos=True)
    genome = _RecordingGenome()
    ex = {"chrom": "1", "start": 1000, "end": 1100}  # midpoint 1050
    out = transform_window_embedding(ex, tok, genome, window_size=20, strand="+")
    # centered: ctx_start = 1050 - 20//2 = 1040, end 1060
    assert genome.calls[-1] == ("1", 1040, 1060, "+")
    assert out["input_ids"].tolist() == tok.encode("A" * 20)


def test_transform_window_embedding_strand_changes_sequence():
    tok = create_char_tokenizer(bos=True, eos=True)
    genome = _RecordingGenome()
    ex = {"chrom": "1", "start": 1000, "end": 1100}
    fwd = transform_window_embedding(ex, tok, genome, window_size=20, strand="+")
    rc = transform_window_embedding(ex, tok, genome, window_size=20, strand="-")
    assert genome.calls[-1][3] == "-"
    assert not torch.equal(fwd["input_ids"], rc["input_ids"])


def test_center_token_bounds_strands_pool_same_genomic_block():
    # W - n = 155 is ODD, so the FWD and RC center starts differ by one — chosen
    # so both strands pool the SAME genomic block (the fix for the strand shift).
    W, n, npfx = 255, 100, 1
    lo_f, hi_f = _center_token_bounds(W, n, npfx, "+")
    lo_r, hi_r = _center_token_bounds(W, n, npfx, "-")
    assert hi_f - lo_f == n and hi_r - lo_r == n
    f0, f1 = lo_f - npfx, hi_f - npfx  # FWD window-coord DNA block [77, 177)
    r0, r1 = lo_r - npfx, hi_r - npfx  # RC  window-coord DNA block [78, 178)
    assert (f0, f1) == (77, 177) and (r0, r1) == (78, 178)
    # RC window index k maps to forward index W-1-k; map the RC block back to
    # forward coords and require it to equal the forward block.
    rc_back = (W - 1 - (r1 - 1), W - 1 - r0 + 1)
    assert rc_back == (f0, f1)


def test_center_token_bounds_even_difference_coincides():
    # W - n even (256-100=156): FWD and RC starts coincide exactly.
    assert _center_token_bounds(256, 100, 1, "+") == _center_token_bounds(
        256, 100, 1, "-"
    )
    # window_size == n_center_bp: whole window pooled, identical on both strands.
    assert _center_token_bounds(100, 100, 1, "+") == _center_token_bounds(
        100, 100, 1, "-"
    )
