"""Per-token validation-loss cache for the stratified-LL-gap analysis (issue #296).

**Stage 1 of #296.** Run a forward pass over a mixed-case validation-interval
dataset and emit the model's per-token loss (``−log p`` of the true base) for
*every* base, in long format keyed by genomic coordinate, with **no bucketing
baked in**. A later CPU pass (Stage 2) annotates each position (codon position,
splice distance, conservation, …) and computes the conserved/non-conserved LL
gap *within* arbitrary strata — without ever re-running the model.

This reuses the LL-gap machinery wholesale: ``transform_ll_clm`` (uppercase
before tokenizing + a source-aligned case mask) and a per-token sibling of
``compute_ll_clm`` (``compute_per_token_ll_clm``, the un-summed ``log p``). Only
the aggregation differs — here we keep *every* token instead of collapsing into
two case buckets. FWD strand only, matching ``compute_hf_ll_gap``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.transforms import _get_special_token_counts, transform_ll_clm
from marin_dna.model.runner import run_inference
from marin_dna.model.scoring import compute_per_token_ll_clm


def compute_hf_per_token_loss(
    checkpoint_path: str | Path,
    sequences: pd.DataFrame,
    window_size: int,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    torch_compile: bool = False,
) -> pd.DataFrame:
    """Per-token loss atoms for an HF causal LM over a mixed-case interval set.

    Loads the checkpoint and runs ``compute_per_token_ll_clm`` (FWD) over the
    mixed-case ``seq`` column through the same Trainer harness as
    ``compute_hf_ll_gap``. Returns one **row per (window, base position)** — long
    format — so Stage 2 can join arbitrary per-position annotations and compute
    a conserved/non-conserved LL gap restricted to any stratum.

    The ``window_size`` DNA bases map 1:1 onto the ``[N, window_size]`` per-token
    output **only when the tokenizer prepends exactly one BOS and appends no
    suffix** — the 255+BOS regime these models train in. With ``input_ids =
    [BOS, seq0..seq_{W-1}]`` the per-token log-prob ``logp[:, j] = log
    p(seq[j] | BOS, seq[<j])`` for every ``j ∈ [0, W)``, so column ``j`` is the
    loss for ``seq[j]`` and ``pos_in_window = j``. A different special-token
    layout (no BOS, an appended EOS, …) shifts that alignment, so it is asserted
    rather than silently mis-binned (CLAUDE.md: fail fast on silent-corruption
    risks).

    Args:
        checkpoint_path: Local HF checkpoint dir (``config.json`` + weights +
            tokenizer).
        sequences: DataFrame with a mixed-case ``seq`` column **and** an ``id``
            column (``chrom:start-end``, 0-based half-open); row order preserved.
            ``id`` is required here (unlike ``compute_hf_ll_gap``) because Stage 2
            needs the per-window coordinate to place each token in the genome.
        window_size: Expected DNA length of every ``seq`` (e.g. 255). Asserted.
        batch_size: Per-device eval batch size.
        num_workers: Dataloader workers.
        torch_compile: Whether to ``torch.compile`` the forward pass.

    Returns:
        Long DataFrame ``[window_id, pos_in_window, loss, ref_base, is_upper]``
        with ``len(sequences) * window_size`` rows, row-major over
        ``(window, position)``. ``loss`` = ``−log p(base)`` in nats (> 0; smaller
        = more predictable). ``pos_in_window`` ∈ ``[0, window_size)``; the
        genomic position of a row is ``start + pos_in_window`` (Stage 2 parses
        ``id``). ``ref_base`` is the uppercased true base; ``is_upper`` is the
        phyloP-conservation bit straight from the source case.
    """
    checkpoint_path = Path(checkpoint_path)
    for col in ("seq", "id"):
        assert col in sequences.columns, (
            f"sequences missing '{col}' column; got {list(sequences.columns)}"
        )
    n = len(sequences)
    assert n > 0, "empty sequences frame"
    seq_lens = sequences["seq"].str.len()
    assert (seq_lens == window_size).all(), (
        f"every seq must be window_size={window_size} bp; got lengths "
        f"{sorted(seq_lens.unique())[:5]}"
    )

    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )

    # Per-token ↔ base alignment requires exactly one prefix (BOS) and no suffix
    # so that logp[:, j] is the loss for seq[j]. Assert loudly — a different
    # layout would shift every position by a constant and silently mis-bin.
    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert (n_prefix, n_suffix) == (1, 0), (
        f"per-token cache assumes 1 BOS + 0 suffix (255+BOS regime); got "
        f"(n_prefix, n_suffix)=({n_prefix}, {n_suffix}). A different special-token "
        "layout shifts the position↔base alignment and must be handled explicitly."
    )

    hf_dataset = Dataset.from_pandas(sequences[["seq"]], preserve_index=False)
    pred = run_inference(
        model,
        tokenizer,
        hf_dataset,
        compute_fn=compute_per_token_ll_clm,
        data_transform_fn=transform_ll_clm,
        data_transform_on_the_fly=True,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": True,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )

    pred = np.asarray(pred, dtype=np.float64)
    # Defensive reshape mirroring compute_hf_ll_gap: some Trainer/device combos
    # have flattened [N, W] to [N*W]. Row-major reshape preserves order.
    if pred.ndim == 1 and pred.shape[0] == n * window_size:
        print(
            f"[per_token_loss] WARNING: inference returned flat shape {pred.shape}; "
            f"reshaping to ({n}, {window_size})."
        )
        pred = pred.reshape(n, window_size)
    assert pred.shape == (n, window_size), (
        f"per-token pred shape {pred.shape}, expected ({n}, {window_size})"
    )
    assert np.isfinite(pred).all(), "non-finite per-token log p"

    # Reconstruct ref base + conservation bit from the source seqs (same order).
    ids = sequences["id"].to_numpy()
    char = np.array([list(s) for s in sequences["seq"]])  # [N, W] of single chars
    assert char.shape == (n, window_size), char.shape
    ref_base = np.char.upper(char)
    is_upper = np.char.isupper(char)

    out = pd.DataFrame(
        {
            "window_id": np.repeat(ids, window_size),
            "pos_in_window": np.tile(np.arange(window_size, dtype=np.int32), n),
            "loss": -pred.reshape(-1),  # −log p, in nats (> 0)
            "ref_base": ref_base.reshape(-1),
            "is_upper": is_upper.reshape(-1),
        }
    )
    # log_softmax is strictly < 0 for finite logits, so −log p > 0; a violation
    # would mean a sign/alignment slip upstream.
    assert (out["loss"] > 0).all(), "non-positive loss — log p ≥ 0?"
    assert out["ref_base"].isin(list("ACGTN")).all(), (
        f"unexpected ref base(s): {sorted(set(out['ref_base']) - set('ACGTN'))}"
    )
    return out
