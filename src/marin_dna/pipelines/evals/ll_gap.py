"""Functional/non-functional LL-gap eval for HF causal LMs (issue #274).

The **LL gap** is the mean log-likelihood the model assigns to uppercase
(phyloP-functional / conserved) target tokens minus the mean LL on lowercase
(non-functional) target tokens, over a mixed-case validation-interval dataset.
A positive gap means functional bases are easier to predict — a self-supervised
proxy for "captures functional vs non-functional sequence structure" (issue #8).

This is the model-agnostic core: the computation kernel
(``marin_dna.model.runner.run_ll_clm`` → ``compute_ll_clm`` over
``transform_ll_clm``) is shared with the Evo2 path; only the model loader
differs. ``compute_hf_ll_gap`` loads an HF ``AutoModelForCausalLM`` directly (no
Evo2 shims). ``aggregate_ll_gap`` (used by both paths) lives here because it is
pure numpy and not Evo2-specific; ``pipelines.evals.evo2`` re-exports it for
backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.model.runner import run_ll_clm


def compute_hf_ll_gap(
    checkpoint_path: str | Path,
    sequences: pd.DataFrame,
    window_size: int,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    torch_compile: bool = False,
) -> pd.DataFrame:
    """Per-sequence functional/non-functional LL atoms for an HF causal LM.

    Loads the checkpoint and runs ``run_ll_clm`` (FWD strand) over the mixed-case
    ``seq`` column. ``transform_ll_clm`` uppercases before tokenizing (the model
    only ever sees uppercase, as in training), prepends BOS if the tokenizer
    defines it, and carries an ``is_upper`` mask aligned to source positions;
    ``compute_ll_clm`` slices that mask to the causal-shifted target positions
    and buckets the per-token log-probs into upper/lower sums + counts.

    Returns per-row **sums and counts**, not means — an all-upper or all-lower
    row would divide by zero. Aggregate across rows with :func:`aggregate_ll_gap`.

    Args:
        checkpoint_path: Local HF checkpoint dir (``config.json`` + weights +
            tokenizer). ``AutoModelForCausalLM`` / ``AutoTokenizer`` satisfy the
            duck-typed interface ``marin_dna.model.runner`` expects.
        sequences: DataFrame with a mixed-case ``seq`` column (and optionally an
            ``id`` column, carried through). Row order is preserved.
        window_size: Expected DNA length of every ``seq`` (the model's training
            context, e.g. 255 for 255+BOS runs). Asserted, to catch a context
            mismatch before an expensive forward pass.
        batch_size: Per-device eval batch size.
        num_workers: Dataloader workers.
        torch_compile: Whether to ``torch.compile`` the forward pass.

    Returns:
        DataFrame with columns ``[id?, ll_sum_upper, ll_sum_lower, n_upper,
        n_lower]`` — one row per input sequence, same order. ``ll_sum_*`` are
        summed ``log p`` over target tokens of that case (negative; closer to 0
        is better); ``n_*`` are the target-token counts.
    """
    checkpoint_path = Path(checkpoint_path)
    assert "seq" in sequences.columns, (
        f"sequences missing 'seq' column; got {list(sequences.columns)}"
    )
    n = len(sequences)
    assert n > 0, "empty sequences frame"
    seq_lens = sequences["seq"].str.len()
    assert (seq_lens == window_size).all(), (
        f"every seq must be window_size={window_size} bp; got lengths "
        f"{sorted(seq_lens.unique())[:5]}"
    )

    # AutoTokenizer / AutoModelForCausalLM satisfy the duck-typed interface
    # marin_dna.model.runner expects — no adapter wrappers needed.
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )

    hf_dataset = Dataset.from_pandas(sequences[["seq"]], preserve_index=False)
    pred = run_ll_clm(
        model,
        tokenizer,
        hf_dataset,
        data_transform_on_the_fly=True,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": True,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )

    pred = np.asarray(pred)
    # Defensive reshape: on some HF Trainer / device combinations run_ll_clm has
    # returned a flat [N*4] array instead of [N, 4] (see compute_evo2_ll). Row-
    # major reshape preserves per-row order.
    if pred.ndim == 1 and pred.shape[0] == n * 4:
        print(
            f"[ll_gap] WARNING: run_ll_clm returned flat shape {pred.shape}; "
            f"reshaping to ({n}, 4). If this recurs, investigate the gather path."
        )
        pred = pred.reshape(n, 4)
    assert pred.shape == (n, 4), (
        f"LL pred shape mismatch: got {pred.shape}, expected ({n}, 4)"
    )
    assert np.isfinite(pred).all(), "non-finite values in HF LL prediction"

    out = pd.DataFrame(
        {
            "ll_sum_upper": pred[:, 0].astype(np.float64),
            "ll_sum_lower": pred[:, 1].astype(np.float64),
            "n_upper": pred[:, 2].astype(np.int64),
            "n_lower": pred[:, 3].astype(np.int64),
        }
    )
    if "id" in sequences.columns:
        out.insert(0, "id", sequences["id"].to_numpy())
    return out


def aggregate_ll_gap(pred: np.ndarray) -> dict[str, float]:
    """Collapse a ``[N, 4]`` per-row LL prediction into dataset-wide
    token-weighted means and the LL gap.

    Cast to fp64 *before* summing — fp32 accumulation drift over ~10^6
    target tokens is non-trivial.

    Args:
        pred: ``[N, 4]`` of ``(ll_sum_upper, ll_sum_lower, n_upper, n_lower)``.

    Returns:
        Dict with ``LL_all``, ``LL_upper``, ``LL_lower``, ``gap``,
        ``n_upper``, ``n_lower``. ``LL_*`` are mean log-likelihoods per
        target token (negative; closer to 0 is better — ``compute_ll_clm``
        returns raw ``log p``, not NLL). ``gap = LL_upper - LL_lower``,
        positive when uppercase (functional) bases are easier to predict
        than lowercase.
    """
    pred = np.asarray(pred)
    assert pred.ndim == 2 and pred.shape[1] == 4, (
        f"expected [N, 4] pred, got {pred.shape}"
    )
    S_u, S_l, n_u, n_l = pred.astype(np.float64).sum(axis=0)
    assert n_u > 0, "no upper (functional) target tokens — check case mask"
    assert n_l > 0, "no lower (non-functional) target tokens — check case mask"
    LL_upper = float(S_u / n_u)
    LL_lower = float(S_l / n_l)
    LL_all = float((S_u + S_l) / (n_u + n_l))
    return {
        "LL_all": LL_all,
        "LL_upper": LL_upper,
        "LL_lower": LL_lower,
        "gap": LL_upper - LL_lower,
        "n_upper": int(n_u),
        "n_lower": int(n_l),
    }
