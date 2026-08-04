"""Preprocessing helpers for wandb eval / lm_eval history series.

Scaling-sweep runs occasionally resume from a checkpoint that's a few
training steps behind the last-logged eval. The resumed job re-emits an
eval at or near the resumed step before continuing, producing two (or
sometimes three) rows that represent the same model state.

Two distinct symptoms appear in the wandb history:

  (a) "Close steps" — the resumed eval is logged within a small number of
      training steps of the original, e.g. Δ=21 at end-of-training final
      evals or Δ=1 inside a single-step restart loop.

  (b) "Identical values" — the resumed eval, being the same checkpoint, is
      logged with a value that is bit-identical to the original entry, even
      when the step counters differ by hundreds (e.g. 134720 → 135050 with
      auprc 0.1729 / 0.1729).

A step-only threshold catches (a) but misses (b) whenever the resume offset
exceeds the threshold (some restarts re-emit the eval ~500 steps later, well
above any safe step-only cutoff that doesn't false-positive on legitimate
eval cadence variations).

`dedup_eval_history` collapses single-linkage clusters detected by EITHER
criterion. It keeps the last entry of each cluster (the one that continues
the forward training trajectory).
"""

from __future__ import annotations

import numpy as np

# Step threshold expressed as a fraction of the run's total training steps.
# 0.1% (≈216 steps for the 215.6k-step scaling sweep) is enough to catch the
# end-of-training Δ=21 final-eval pair and short Δ≤200 resume offsets; longer
# resume offsets are caught by the value-equality criterion.
DEFAULT_MAX_GAP_FRACTION: float = 0.001

# Tolerance for "values are the same checkpoint." Resumed evals come from
# bit-identical model state and serialize identically through wandb, so any
# reasonable atol works; we use a tight value to make this near-impossible to
# false-positive on two truly independent evals that happen to look close.
DEFAULT_VALUE_ATOL: float = 1e-9


def dedup_eval_history(
    steps: np.ndarray,
    values: np.ndarray,
    max_gap_steps: float,
    value_atol: float = DEFAULT_VALUE_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse single-linkage clusters of checkpoint-resume duplicates.

    Two consecutive entries (sorted by step ascending) are linked into the
    same cluster when EITHER:
      - their step gap is ≤ `max_gap_steps`, OR
      - their values are equal within `value_atol`.

    Each cluster collapses to its last entry. This is robust to step-
    threshold misspecification: a long resume offset (e.g. 562 steps at
    128M, 308 at 1B) is still caught because the resumed eval is value-
    identical to the original. Conversely, a legitimately close eval pair
    with distinct values (e.g. the end-of-training Δ=21 pair) is still
    caught by the step criterion even when values differ.

    Inputs are assumed sorted by step ascending and the same length.
    """
    steps = np.asarray(steps)
    values = np.asarray(values, dtype=float)
    if len(steps) <= 1:
        return steps, values
    diffs = np.diff(steps)
    step_close = diffs <= max_gap_steps
    val_equal = np.isclose(values[:-1], values[1:], rtol=0.0, atol=value_atol)
    # Entry i (0..n-2) belongs to the same cluster as i+1 iff either linkage
    # criterion holds; in that case drop entry i (its cluster's last entry
    # will be kept further along). The final entry is always kept.
    in_chain = step_close | val_equal
    keep = np.ones(len(steps), dtype=bool)
    keep[:-1] = ~in_chain
    return steps[keep], values[keep]


# Back-compat alias for the previous single-criterion entrypoint.
collapse_close_steps = dedup_eval_history
