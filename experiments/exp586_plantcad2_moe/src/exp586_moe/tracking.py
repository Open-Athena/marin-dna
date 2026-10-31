"""Monotonic W&B history for diagnostic attempts sharing one run identity."""

import jax
import numpy as np
import wandb
from jax.experimental import multihost_utils
from levanter import tracker


def wandb_history_offset() -> int:
    """Query server history before logging; local Run.step can reset on resume.

    Call once after tracker initialization, before any history writes. Callers add
    positive local steps to this offset. Broadcast failure before raising so all
    ranks fail together instead of silently reusing old history step numbers.
    """
    offset = -1
    error = None
    if jax.process_index() == 0:
        try:
            run = tracker.get_tracker("wandb").run
            public = wandb.Api(timeout=30).run(f"{run.entity}/{run.project}/{run.id}")
            offset = max(0, int(run.step), int(public.lastHistoryStep))
            if offset > np.iinfo(np.int32).max - 1001:
                raise ValueError("W&B step exceeds bounded diagnostic counter range")
        except Exception as caught:  # noqa: BLE001 -- broadcast before raising on every rank
            offset = -1
            error = caught
    offset = int(
        multihost_utils.broadcast_one_to_all(np.asarray(offset, dtype=np.int32))
    )
    if offset < 0:
        raise RuntimeError("Cannot establish the W&B history offset") from error
    tracker.log_summary({"validation/wandb_history_offset": offset})
    return offset
