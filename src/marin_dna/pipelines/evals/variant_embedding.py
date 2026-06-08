"""Per-variant ref/alt hidden-state embeddings (issue #302 embedding probe).

For each SNV, build the same centered ``window_size`` context used by the LLR
scoring (``_get_variant_window`` — variant at ``in_seq_var_pos``), then read the
model's hidden states for the **ref** window and the **alt** window (alt = ref
with the single center token swapped to the alt nucleotide). For a chosen set of
layers, mean-pool the center ``n_center_bp`` token positions (strand-aware, via
``_center_token_bounds``) and average the forward + reverse-complement strands —
the #246 ``run_window_embeddings`` recipe, extended to (a) the alt allele and (b)
several layers in one pass.

Returns per-variant ``ref`` and ``alt`` pooled embeddings of shape
``[N, n_layers, D]`` (FWD+RC averaged). Downstream a probe forms ``concat[ref,
alt]`` or ``delta = alt − ref`` per layer. The point of the multi-layer read is
that the *last* layer is specialized for the next-token (likelihood) readout,
while a *middle* layer may separate pathogenic-vs-benign better — testing whether
the missense degradation lives in the representation or only in the readout.

Coordinates are 1-based VCF ``pos`` (as the rest of ``transforms`` expects);
sequence extraction is 0-based half-open inside ``Genome``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from marin_dna.data.dna import complement_base
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    _get_variant_window,
    in_seq_var_pos,
)
from marin_dna.model.runner import _center_token_bounds


def resolve_layer_indices(
    num_hidden_layers: int, layer_fracs: tuple[float, ...]
) -> list[int]:
    """Fractional depths → indices into the HF ``hidden_states`` tuple.

    ``hidden_states`` has ``num_hidden_layers + 1`` entries (index 0 = embeddings,
    ``k`` = output of layer ``k``; ``num_hidden_layers`` == last == ``-1``). A
    fraction ``f`` maps to ``round(f * num_hidden_layers)``, clamped to
    ``[1, num_hidden_layers]`` and de-duplicated.
    """
    idx = sorted(
        {
            min(num_hidden_layers, max(1, round(f * num_hidden_layers)))
            for f in layer_fracs
        }
    )
    return idx


@torch.no_grad()
def compute_variant_embeddings(
    model: Any,
    tokenizer: Any,
    genome: Any,
    variants: list[dict[str, Any]],
    window_size: int,
    *,
    layer_indices: list[int],
    n_center_bp: int = 100,
    rc: bool = True,
    batch_size: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[np.ndarray, np.ndarray]:
    """Ref/alt center-pooled, FWD+RC-averaged embeddings at several layers.

    Args:
        model: HF model exposing ``model(input_ids, output_hidden_states=True)``
            with a ``.hidden_states`` tuple (e.g. ``AutoModel`` or
            ``AutoModelForCausalLM``).
        tokenizer: HF tokenizer (``.encode``, ``.bos_token_id``).
        genome: ``marin_dna.data.genome.Genome``.
        variants: list of ``{"chrom","pos","ref","alt"}`` (1-based ``pos``).
        window_size: DNA context length (e.g. 255).
        layer_indices: indices into the ``hidden_states`` tuple to pool.
        n_center_bp: number of central DNA positions to mean-pool.
        rc: also score the reverse-complement window and average.
        batch_size, device, dtype: inference knobs.

    Returns:
        ``(ref_emb, alt_emb)``, each ``[N, len(layer_indices), D]`` float32.
    """
    model = model.to(device=device, dtype=dtype).eval()
    n_prefix, _ = _get_special_token_counts(tokenizer)
    nuc_ids = _get_nucleotide_token_ids(tokenizer)
    n = len(variants)
    n_layers = len(layer_indices)

    def _strand(strand: str) -> tuple[np.ndarray, np.ndarray]:
        var_pos = (
            in_seq_var_pos(window_size, strand) + n_prefix
        )  # token index of the variant
        tok_lo, tok_hi = _center_token_bounds(
            window_size, n_center_bp, n_prefix, strand
        )
        ref_out: np.ndarray | None = None
        alt_out: np.ndarray | None = None
        for s in range(0, n, batch_size):
            chunk = variants[s : s + batch_size]
            ids, alt_tok = [], []
            for v in chunk:
                seq, _ = _get_variant_window(v, genome, window_size, strand)
                ids.append(tokenizer.encode(seq))
                alt = v["alt"] if strand == "+" else complement_base(v["alt"])
                alt_tok.append(nuc_ids[alt])
            input_ids = torch.tensor(ids, dtype=torch.long, device=device)
            assert input_ids.shape[1] == window_size + n_prefix, (
                f"unexpected token length {input_ids.shape[1]} (want {window_size + n_prefix})"
            )
            # sanity: the token at var_pos is the ref nucleotide token
            ref_tok = torch.tensor(
                [
                    nuc_ids[v["ref"] if strand == "+" else complement_base(v["ref"])]
                    for v in chunk
                ],
                device=device,
            )
            assert torch.equal(input_ids[:, var_pos], ref_tok), (
                "ref token mismatch at var_pos"
            )
            alt_ids = input_ids.clone()
            alt_ids[:, var_pos] = torch.tensor(alt_tok, dtype=torch.long, device=device)

            hs_ref = model(input_ids=input_ids, output_hidden_states=True).hidden_states
            hs_alt = model(input_ids=alt_ids, output_hidden_states=True).hidden_states
            if ref_out is None:
                d = hs_ref[0].shape[-1]
                ref_out = np.empty((n, n_layers, d), dtype=np.float32)
                alt_out = np.empty((n, n_layers, d), dtype=np.float32)
            for li, layer in enumerate(layer_indices):
                ref_out[s : s + len(chunk), li] = (
                    hs_ref[layer][:, tok_lo:tok_hi].mean(1).float().cpu().numpy()
                )
                alt_out[s : s + len(chunk), li] = (
                    hs_alt[layer][:, tok_lo:tok_hi].mean(1).float().cpu().numpy()
                )
        assert ref_out is not None and alt_out is not None
        return ref_out, alt_out

    ref_f, alt_f = _strand("+")
    if not rc:
        return ref_f, alt_f
    ref_r, alt_r = _strand("-")
    return (ref_f + ref_r) / 2, (alt_f + alt_r) / 2
