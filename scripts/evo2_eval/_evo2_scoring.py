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


def fwd_rc_average_f16(
    strand_embs: list[np.ndarray], out_dtype: np.typing.DTypeLike = np.float16
) -> np.ndarray:
    """FWD+RC mean of per-strand fp32 pooled embeddings, cast to ``out_dtype`` for storage.

    Each ``strand_embs[i]`` is one strand's pooled vectors ``[N, K]`` (fp32). The
    mean is the #314 protocol's FWD+RC average. Accumulate in **fp32** and cast to
    the storage dtype only at the very end: f16 storage of allele *means* already
    bounds the downstream probe feature ``delta = emb_alt - emb_ref`` (a small
    difference of two near-equal vectors — catastrophic cancellation), so the
    aggregation itself must not add rounding on top of it (issue #318). Returns
    ``[N, K]`` in ``out_dtype``.

    ``out_dtype`` defaults to ``float16`` — the gLM #325 schema, half the storage,
    and validated adequate for *that* model family's activations. Evo2 is known for
    persistent "massive-activation" channels whose post-norm magnitude can give f16
    coarse resolution exactly where the delta cancels the shared context (issue #131);
    ``float32`` is the lossless escape hatch (2× storage, same probe-compatible
    ``list`` column — ``pair_feature_from_bundle`` upcasts to f32 either way). The
    ``embed_smoke.py`` check measures the f16-vs-f32 delta corruption to decide.

    Duplicated from ``marin_dna.pipelines.evals.inference.fwd_rc_average_f16`` (the
    gLM #325 driver), with the ``out_dtype`` param added: the Evo2 docker image
    deliberately omits transformers, so importing that module (top-level
    ``from transformers import ...``) would fail. The helper is pure-numpy and
    model-agnostic; per CLAUDE.md this is the "duplication beats a cross-dependency"
    case.
    """
    assert strand_embs, "need at least one strand's embeddings to average"
    acc = np.zeros_like(strand_embs[0], dtype=np.float32)
    for e in strand_embs:
        acc += np.asarray(e, dtype=np.float32)
    acc /= len(strand_embs)
    out = acc.astype(out_dtype)
    # Fail loud rather than silently ship inf: a pooled mean beyond float16's
    # ±65504 (e.g. a persistent massive-activation channel) overflows on the cast,
    # which would NaN-poison the downstream delta = emb_alt - emb_ref. (No-op for
    # float32, whose range covers any finite fp32 mean — a free safety check.)
    assert np.isfinite(out).all(), (
        f"non-finite pooled embedding after {np.dtype(out_dtype).name} cast — the "
        f"fp32 mean exceeded the storage dtype's range or was non-finite upstream"
    )
    return out


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
    return_embeddings: bool = False,
    emb_layer: str | None = None,
) -> torch.Tensor:
    """Return ``[B, 2] = (LLR, next_token_jsd_mean)`` for a batch.

    No KV-cache: feeds the full ref and alt sequences (``[2B, L]``) through
    the model in one pass, then extracts LLR and JSD from the per-position
    4-nuc softmax. Mirrors the math of ``marin_dna.model.scoring.compute_variant_score_bundle``
    but without prefix sharing.

    Args:
        evo2_model: An ``evo2.Evo2`` instance. Native API used directly:
            ``outputs, emb = evo2_model(input_ids, ...)`` with ``outputs[0]`` as
            logits and ``emb`` a ``{layer_name: tensor}`` dict (empty unless
            ``return_embeddings`` is passed to Evo2).
        input_ids: Ref token IDs ``[B, L]``.
        alt_token_id: Alt nucleotide token ID per row ``[B]``.
        var_pos: Variant position (Python int, constant within batch).
        nuc_token_ids: Token IDs for [A, C, G, T] in tokenizer order ``[4]``.
        return_embeddings: If True, also emit the entire-window mean-pooled,
            both-allele hidden states from the layer named by ``emb_layer``,
            captured from the **same single** ``[2B, L]`` forward that produces the
            logits (no second pass — mirrors the gLM #325 bundle). Returns
            ``[B, 2 + 2D]`` instead of ``[B, 2]``.
        emb_layer: Evo2 module name to read the hidden state from (e.g. ``"norm"``
            for the post-final-norm last-layer state — the HF ``last_hidden_state``
            analog). Required when ``return_embeddings``. Passed straight to Evo2's
            ``layer_names``; validated against ``evo2_model.model.named_modules()``
            by the driver before the loop.

    Returns:
        ``[B, 2]`` of ``(LLR, next_token_jsd_mean)``, or — when
        ``return_embeddings`` — ``[B, 2 + 2D]`` with columns ``[0:2]`` the two
        scores, ``[2:2+D]`` the pooled ref embedding, ``[2+D:2+2D]`` the pooled
        alt embedding (``D`` = model hidden size). Embeddings are fp32 (the driver
        FWD+RC-averages and casts to f16).
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
        if return_embeddings:
            assert emb_layer is not None, "emb_layer required when return_embeddings"
            # Evo2's native hook path: registers a forward hook on the named module
            # and returns its output in `emb` — the same single forward also yields
            # the logits, so no second pass.
            outputs, emb = evo2_model(
                combined, return_embeddings=True, layer_names=[emb_layer]
            )
            hidden = emb[emb_layer]  # [2B, L, D]
        else:
            outputs, _ = evo2_model(combined)
            hidden = None
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

    scores = torch.stack([llr, jsd_mean], dim=1)  # [B, 2]
    if not return_embeddings:
        return scores

    # Entire-window mean-pool of the chosen layer's hidden states, ref & alt.
    # Evo2 has no BOS/special tokens (CharLevelTokenizer), so the input IS the
    # window_size DNA window — pool over all L positions ([0, L)), the literal
    # #314 "entire_window". `hidden` is [2B, L, D] in call order [ref rows; alt
    # rows] (combined = cat([ref, alt])); reshape to [2, B, L, D] and reduce L.
    # Accumulate in fp32 (bf16/fp8 summed over thousands of positions accrues
    # rounding error) via sum(dtype=float32)/L — bit-identical to .float().mean()
    # but skips a full fp32 copy of the [2B, L, D] block. The driver FWD+RC
    # -averages (still fp32) and casts to f16.
    assert hidden is not None
    assert hidden.shape[:2] == (2 * B, L), (
        f"hidden {tuple(hidden.shape)} != expected (2B={2 * B}, L={L}, D); "
        f"emb_layer={emb_layer!r} captured an unexpected shape"
    )
    d = hidden.shape[-1]
    # reshape (not view): a hook-captured hidden state may be non-contiguous. The
    # leading 2B rows are ordered [ref (B); alt (B)] (combined = cat([ref, alt])),
    # so (2, B) splits ref/alt correctly.
    pooled = hidden.reshape(2, B, L, d).sum(dim=2, dtype=torch.float32) / L  # [2, B, D]
    emb_ref = pooled[0]  # [B, D]
    emb_alt = pooled[1]  # [B, D]
    return torch.cat([scores.float(), emb_ref, emb_alt], dim=1)  # [B, 2 + 2D]


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
    return_embeddings: bool = False,
    emb_layer: str = "norm",
    emb_dtype: str = "float16",
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
            ``return_embeddings`` makes the forward heavier (it materializes a
            ``[2*batch_size, window_size, D]`` hidden state) — use a smaller batch.
        rc_avg: If True, score forward + reverse-complement windows and
            return the element-wise mean (matches evals_v2 protocol).
        return_embeddings: If True, also emit the entire-window mean-pooled,
            both-allele, FWD+RC-averaged last-layer embeddings as ``emb_ref`` /
            ``emb_alt`` ``list[f16]`` columns — the same schema the gLM #325
            bundle writes, so the #320 linear probe + #331 per-chrom AUPRC consume
            them unchanged. Requires ``rc_avg=True`` (the stored vector is the
            FWD+RC average; a forward-only embedding would be silently mislabeled).
        emb_layer: Evo2 module name to pool the hidden state from when
            ``return_embeddings`` (default ``"norm"`` = post-final-norm last-layer
            state, the HF ``last_hidden_state`` analog). Validated against
            ``evo2.model.named_modules()`` before the run; fails loud with the
            candidate names if absent.
        emb_dtype: Storage dtype for ``emb_ref``/``emb_alt`` — ``"float16"``
            (default, the gLM #325 schema) or ``"float32"`` (lossless escape hatch
            for Evo2's massive-activation channels, #131). The pool + FWD+RC average
            are always fp32; this only sets the final cast. The probe upcasts to f32
            regardless, so both are probe-compatible.

    Returns:
        DataFrame with [llr, minus_llr, abs_llr, next_token_jsd_mean] (× {avg,
        _fwd, _rev} when ``rc_avg``). With ``return_embeddings``, also ``emb_ref``
        and ``emb_alt`` (each a length-``D`` f16 vector per row).
    """
    from evo2 import Evo2

    assert rc_avg or not return_embeddings, (
        "return_embeddings=True requires rc_avg=True — the stored embedding is the "
        "FWD+RC average; a forward-only embedding would be silently mislabeled"
    )
    assert emb_dtype in ("float16", "float32"), (
        f"emb_dtype must be 'float16' or 'float32', got {emb_dtype!r}"
    )

    fa = Fasta(str(genome_path))
    evo2 = Evo2(model_name)
    tokenizer = evo2.tokenizer
    # Resolve device from the model.
    try:
        device = next(evo2.model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cuda:0")

    if return_embeddings:
        # Fail fast with the candidate names rather than letting Evo2's hook path
        # silently capture nothing (empty emb dict → KeyError mid-loop). Evo2 matches
        # `layer_names` against `evo2.model.named_modules()`.
        available = {name for name, _ in evo2.model.named_modules()}
        assert emb_layer in available, (
            f"emb_layer={emb_layer!r} is not a module in evo2.model; "
            f"final-stack candidates: "
            f"{sorted(n for n in available if 'norm' in n or n.endswith('unembed'))} "
            f"(blocks: {sum(1 for n in available if n.startswith('blocks.') and n.count('.') == 1)})"
        )
        print(f"[evo2] return_embeddings=True, emb_layer={emb_layer!r}", flush=True)

    nuc_token_ids = torch.tensor(
        [int(tokenizer.tokenize(b)[0]) for b in "ACGT"], dtype=torch.long, device=device
    )

    n = len(df)
    strands: tuple[Literal["+", "-"], ...] = ("+", "-") if rc_avg else ("+",)
    # strand → [N, W] kernel output, W = 2 (scores) or 2 + 2D (scores + emb_ref/alt).
    per_strand: dict[str, np.ndarray] = {}
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
        strand_out: np.ndarray | None = None  # allocated on the first batch (width W)
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
                return_embeddings=return_embeddings,
                emb_layer=emb_layer if return_embeddings else None,
            )
            bundle_np = bundle.detach().cpu().numpy()
            if strand_out is None:
                strand_out = np.zeros((n, bundle_np.shape[1]), dtype=np.float64)
            strand_out[i : i + batch_size] = bundle_np
        assert strand_out is not None, "empty df — no batches ran"
        per_strand[strand] = strand_out

    def _expand(out_arr: np.ndarray, suffix: str) -> dict[str, np.ndarray]:
        # out_arr[:, :2] are the two scores even when embeddings ride along.
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

    cols: dict[str, object] = {}
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

    if return_embeddings:
        # Each per-strand array is [N, 2 + 2D]: [:, 2:2+D] = emb_ref, [:, 2+D:] =
        # emb_alt (fp32, kernel-pooled). FWD+RC-average the full [N, 2D] emb block in
        # fp32 then split — the average is linear, so split-then-average ==
        # average-then-split. Store each allele as a length-D f16 vector per row
        # (list[f16] column), matching the gLM #325 schema the probe reads.
        width = per_strand["+"].shape[1]
        d = (width - 2) // 2
        assert width == 2 + 2 * d, (
            f"score-bundle width {width} != 2 + 2*D; embeddings mis-sized"
        )
        avg_emb = fwd_rc_average_f16(
            [per_strand[s][:, 2:] for s in strands], out_dtype=np.dtype(emb_dtype)
        )  # [N, 2D]
        cols["emb_ref"] = list(avg_emb[:, :d])
        cols["emb_alt"] = list(avg_emb[:, d:])
    return pd.DataFrame(cols)
