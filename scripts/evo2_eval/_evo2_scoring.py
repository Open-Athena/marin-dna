"""Script-local Evo2 variant scoring — independent of marin_dna.model.*.

The shared ``marin_dna.model.scoring.compute_variant_score_bundle`` kernel
introduced in PR #184 uses prefix-sharing via KV-cache to halve the suffix
compute, but Evo2's Vortex backend doesn't expose its internal state in
HF-cache format (the model's HF wrapper returns ``SimpleNamespace(logits=...)``
with no ``past_key_values``). That mismatch is structural: trying to wedge
Evo2 into the prefix-sharing path leaks Evo2 quirks back into the main
kernel.

This module reimplements the LLR + next-token JSD scoring without any
KV-cache, using a single batched forward pass over concatenated
ref and alt sequences (``[2B, L]``). Compute is ~1.3-2× the shared kernel
for HF gLMs, but Evo2's per-variant cost is dominated by the model
itself anyway — and Evo2 is a baseline, not a first-class model.

Inputs are dataset rows ``(chrom, pos, ref, alt)`` and a Genome reader.
RC averaging is done at the kernel-output level (numpy mean of two
``[N, 2]`` arrays), matching the convention of the shared runner.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from tqdm import tqdm

from marin_dna.data.dna import complement_base, reverse_complement


def _get_variant_window(
    fa: Fasta,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    window_size: int,
    strand: Literal["+", "-"],
) -> tuple[str, int, str]:
    """Return ``(window_seq, var_pos, alt_in_window)`` for one variant.

    Coordinates: ``pos`` is 1-based (HF dataset convention). The window is
    centered with ``window_size // 2`` bases on the left, so the variant lands
    at token index ``window_size // 2`` on the forward strand. On the RC
    strand the variant lands at ``window_size - 1 - window_size // 2``.
    """
    assert len(ref) == 1 and len(alt) == 1, f"SNV only; got ref={ref!r} alt={alt!r}"
    var_idx = pos - 1
    left = window_size // 2
    start = var_idx - left
    end = start + window_size
    if start < 0 or end > len(fa[chrom]):
        # Chromosome boundary; let pyfaidx pad with N's (returns shorter slice
        # at boundaries — we pad ourselves to be safe).
        slc = str(fa[chrom][max(0, start) : min(end, len(fa[chrom]))]).upper()
        if start < 0:
            slc = "N" * (-start) + slc
        if end > len(fa[chrom]):
            slc = slc + "N" * (end - len(fa[chrom]))
        seq = slc
    else:
        seq = str(fa[chrom][start:end]).upper()
    assert len(seq) == window_size

    if strand == "+":
        var_pos = left
        assert seq[var_pos] == ref, (
            f"ref mismatch at {chrom}:{pos} (FWD): expected {ref!r}, "
            f"got {seq[var_pos]!r} in seq[{var_pos}]"
        )
        return seq, var_pos, alt
    else:
        seq_rc = reverse_complement(seq)
        var_pos = window_size - 1 - left
        ref_rc = complement_base(ref)
        alt_rc = complement_base(alt)
        assert seq_rc[var_pos] == ref_rc, (
            f"ref mismatch at {chrom}:{pos} (RC): expected {ref_rc!r}, "
            f"got {seq_rc[var_pos]!r} in seq_rc[{var_pos}]"
        )
        return seq_rc, var_pos, alt_rc


def _compute_evo2_kernel(
    evo2_model: Any,
    input_ids: torch.Tensor,  # [B, L]
    alt_token_id: torch.Tensor,  # [B]
    *,
    var_pos: int,
    nuc_token_ids: torch.Tensor,  # [4]
) -> torch.Tensor:
    """Return ``[B, 2] = (LLR, next_token_jsd_mean)`` for a batch.

    No KV-cache: feeds the full ref and alt sequences (``[2B, L]``) through
    the model in one pass, then extracts LLR and JSD from the per-position
    4-nuc softmax. Mirrors the math of ``marin_dna.model.scoring.compute_variant_score_bundle``
    but without prefix sharing.

    Args:
        evo2_model: An ``evo2.Evo2`` instance. Native API used directly:
            ``outputs, _ = evo2_model(input_ids)`` with ``outputs[0]`` as logits.
        input_ids: Ref token IDs ``[B, L]``.
        alt_token_id: Alt nucleotide token ID per row ``[B]``.
        var_pos: Variant position (Python int, constant within batch).
        nuc_token_ids: Token IDs for [A, C, G, T] in tokenizer order ``[4]``.
    """
    B, L = input_ids.shape
    p = var_pos
    assert 0 < p < L - 1, f"var_pos={p} outside (0, L={L}-1)"

    # Build alt sequence (functional, no in-place).
    alt_seq = torch.cat(
        [input_ids[:, :p], alt_token_id.unsqueeze(-1), input_ids[:, p + 1 :]], dim=-1
    )
    combined = torch.cat([input_ids, alt_seq], dim=0)  # [2B, L]

    with torch.inference_mode():
        outputs, _ = evo2_model(combined)
    logits = outputs[0]  # [2B, L, V]

    # 4-nuc log_softmax in fp32 (numerical stability — biofoundation #21).
    log_p_nuc = F.log_softmax(logits[..., nuc_token_ids].float(), dim=-1)  # [2B, L, 4]
    log_p_nuc = log_p_nuc.view(2, B, L, 4)  # [ref/alt, B, L, 4]

    # LLR at variant position: prediction at index p-1 predicts token at p.
    # Both ref and alt input sequences share the prefix [0, p), so their
    # logits at position p-1 are identical — use either.
    log_p_at_var = log_p_nuc[0, :, p - 1]  # [B, 4]
    ref_idx = (input_ids[:, p].unsqueeze(-1) == nuc_token_ids).int().argmax(-1)  # [B]
    alt_idx = (alt_token_id.unsqueeze(-1) == nuc_token_ids).int().argmax(-1)  # [B]
    llr_at_var = log_p_at_var.gather(-1, alt_idx.unsqueeze(-1)).squeeze(
        -1
    ) - log_p_at_var.gather(-1, ref_idx.unsqueeze(-1)).squeeze(-1)  # [B]

    # Downstream positions: index k in [p, L-2] predicts token at k+1.
    log_p_ref_ds = log_p_nuc[0, :, p : L - 1]  # [B, L-1-p, 4]
    log_p_alt_ds = log_p_nuc[1, :, p : L - 1]  # [B, L-1-p, 4]
    targets = input_ids[:, p + 1 :]  # [B, L-1-p]
    target_idx = (targets.unsqueeze(-1) == nuc_token_ids).int().argmax(-1)  # [B, L-1-p]
    target_idx_e = target_idx.unsqueeze(-1)
    llr_downstream = (
        log_p_alt_ds.gather(-1, target_idx_e).squeeze(-1)
        - log_p_ref_ds.gather(-1, target_idx_e).squeeze(-1)
    ).sum(dim=-1)  # [B]
    llr = llr_at_var + llr_downstream

    # JSD per downstream position, averaged.
    log_m = torch.logaddexp(log_p_ref_ds, log_p_alt_ds) - math.log(2.0)
    p_ref = log_p_ref_ds.exp()
    p_alt = log_p_alt_ds.exp()
    kl_ref = (p_ref * (log_p_ref_ds - log_m)).sum(dim=-1)  # [B, L-1-p]
    kl_alt = (p_alt * (log_p_alt_ds - log_m)).sum(dim=-1)
    jsd_mean = (0.5 * (kl_ref + kl_alt)).mean(dim=-1)  # [B]

    return torch.stack([llr, jsd_mean], dim=1)


def _build_token_arrays(
    df: pd.DataFrame,
    fa: Fasta,
    tokenizer: Any,
    window_size: int,
    strand: Literal["+", "-"],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tokenize per-variant windows, return ``(input_ids[N, L], alt_token_id[N], var_pos)``."""
    n = len(df)
    input_ids = np.empty((n, window_size), dtype=np.int64)
    alt_token_ids = np.empty(n, dtype=np.int64)
    var_pos_canonical: int | None = None
    nuc_id = {b: int(tokenizer.tokenize(b)[0]) for b in "ACGT"}
    chroms = df["chrom"].to_numpy()
    positions = df["pos"].to_numpy()
    refs = df["ref"].to_numpy()
    alts = df["alt"].to_numpy()
    for i in range(n):
        seq, var_pos, alt_in_window = _get_variant_window(
            fa,
            chroms[i],
            int(positions[i]),
            refs[i],
            alts[i],
            window_size,
            strand,
        )
        if var_pos_canonical is None:
            var_pos_canonical = var_pos
        else:
            assert var_pos == var_pos_canonical, "var_pos drifted across rows"
        tokens = list(map(int, tokenizer.tokenize(seq)))
        assert len(tokens) == window_size, (
            f"tokenizer emitted {len(tokens)} tokens for {window_size}-bp window"
        )
        input_ids[i] = tokens
        alt_token_ids[i] = nuc_id[alt_in_window]
    assert var_pos_canonical is not None
    return input_ids, alt_token_ids, var_pos_canonical


def compute_evo2_bundle(
    model_name: str,
    df: pd.DataFrame,  # cols: chrom, pos, ref, alt
    genome_path: str | Path,
    window_size: int = 8192,
    batch_size: int = 16,
    rc_avg: bool = True,
) -> pd.DataFrame:
    """Score Evo2 variants → DataFrame[llr, minus_llr, abs_llr, next_token_jsd_mean].

    Stand-alone Evo2 path: loads ``evo2.Evo2`` directly, runs without HF
    Trainer, computes the LLR+JSD bundle without KV-cache prefix-sharing.

    Args:
        model_name: One of evo2's model names (e.g. ``evo2_1b_base``).
        df: DataFrame with [chrom, pos, ref, alt]. Output row-aligned.
        genome_path: Local BGZF-compressed FASTA (with .fai and .gzi).
        window_size: Context length (Evo2 design = 8192).
        batch_size: Per-batch row count. With this kernel each batch
            feeds the model ``[2*batch_size, window_size]`` tokens.
        rc_avg: If True, score forward + reverse-complement windows and
            return the element-wise mean (matches evals_v2 protocol).

    Returns:
        DataFrame with [llr, minus_llr, abs_llr, next_token_jsd_mean].
    """
    from evo2 import Evo2

    fa = Fasta(str(genome_path))
    evo2 = Evo2(model_name)
    tokenizer = evo2.tokenizer
    # Resolve device from the model.
    try:
        device = next(evo2.model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cuda:0")

    nuc_token_ids = torch.tensor(
        [int(tokenizer.tokenize(b)[0]) for b in "ACGT"], dtype=torch.long, device=device
    )

    n = len(df)
    strands: tuple[Literal["+", "-"], ...] = ("+", "-") if rc_avg else ("+",)
    per_strand: dict[str, np.ndarray] = {}  # strand → [N, 2] kernel output
    for strand in strands:
        print(f"[evo2] strand={strand}: tokenizing {n} variants...", flush=True)
        input_ids_np, alt_token_ids_np, var_pos = _build_token_arrays(
            df, fa, tokenizer, window_size, strand
        )
        input_ids = torch.from_numpy(input_ids_np).to(device)
        alt_token_ids = torch.from_numpy(alt_token_ids_np).to(device)
        print(
            f"[evo2] strand={strand}: var_pos={var_pos}, "
            f"running inference (bs={batch_size}, {n} variants)...",
            flush=True,
        )
        strand_out = np.zeros((n, 2), dtype=np.float64)
        batch_starts = list(range(0, n, batch_size))
        for i in tqdm(batch_starts, desc=f"strand={strand}", unit="batch"):
            batch_ids = input_ids[i : i + batch_size]
            batch_alt = alt_token_ids[i : i + batch_size]
            bundle = _compute_evo2_kernel(
                evo2,
                batch_ids,
                batch_alt,
                var_pos=var_pos,
                nuc_token_ids=nuc_token_ids,
            )
            strand_out[i : i + batch_size] = bundle.detach().cpu().numpy()
        per_strand[strand] = strand_out

    def _expand(out_arr: np.ndarray, suffix: str) -> dict[str, np.ndarray]:
        llr_ = out_arr[:, 0]
        jsd_ = out_arr[:, 1]
        assert np.isfinite(llr_).all(), f"non-finite LLR{suffix}"
        assert np.isfinite(jsd_).all() and (jsd_ >= 0).all(), (
            f"non-finite or negative JSD{suffix}"
        )
        return {
            f"llr{suffix}": llr_,
            f"minus_llr{suffix}": -llr_,
            f"abs_llr{suffix}": np.abs(llr_),
            f"next_token_jsd_mean{suffix}": jsd_,
        }

    cols: dict[str, np.ndarray] = {}
    if rc_avg:
        # FWD+RC averaging: keep per-strand columns + the averaged columns
        # (which is what the leaderboard scores against). Per-strand columns
        # let downstream callers sanity-check #175's patterns (fwd ≈ rev
        # individually, low correlation, avg > individual for some subsets).
        avg = (per_strand["+"] + per_strand["-"]) / 2
        cols.update(_expand(avg, suffix=""))
        cols.update(_expand(per_strand["+"], suffix="_fwd"))
        cols.update(_expand(per_strand["-"], suffix="_rev"))
    else:
        cols.update(_expand(per_strand["+"], suffix=""))
    return pd.DataFrame(cols)
