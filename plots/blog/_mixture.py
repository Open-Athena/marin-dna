"""Offline data layer for the mixture figures (9/10) — the evals_v2 analog of
Eric's W&B ``data.py`` for the v0.9 mixture sweep.

Eric's Figures 9/10 read *in-training* VEP AUPRC from W&B history. The blog redo
(epic #361) scores the same runs **offline** through evals_v2 → S3, across the four
metric-worlds ({Mendelian, SGE} × {zero-shot LLR, frozen-embedding probe}). This
module is the bridge: it maps each lineage run's **HF-exported checkpoints** (the
only steps offline-reproducible — see #273) to their evals_v2 ``config.yaml`` model
ids, so a world reader (``_worlds.World.read``) can pull per-checkpoint AUPRC.

Two hard constraints distinguish the offline redo from Eric's:

  1. **Coarse trajectory.** Offline can only score the **HF-exported** checkpoints
     (3–4 per stage, back-half-weighted), not W&B's dense in-training evals
     (~10+/stage). Figure 10's composed trajectories are therefore sparser and lean
     harder on the kernel smoother — flag this in the caption, don't pretend to the
     old density.
  2. **Token accounting is reused verbatim.** ``tokens`` / ``num_train_steps`` /
     mixture ``weights`` are training-intrinsic (``_mixture_lineage`` + the vendored
     ``data_mixture_results.csv``), so the fork-fraction / cumulative-token math
     ports unchanged; only the AUPRC *values* come from offline scoring.

``HF_STEPS`` is the ground truth (a ``gcloud storage ls`` of each run's ``hf/``
under ``gs://marin-us-east5/checkpoints`` on 2026-07-08). Keep it in sync with the
evals_v2 ``config.yaml`` mixture entries this module generates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from plots.blog import _mixture_lineage as ml

_GCS_BUCKET = "gs://marin-us-east5/checkpoints"
_WINDOW_SIZE = 255
_DATA = Path(__file__).resolve().parent / "data" / "data_mixture_results.csv"

# TRAINING metadata source (tokens / num_train_steps / state / mixture weights):
# reused verbatim from Eric's run-metadata CSV — run-intrinsic, not eval results.
RESULTS_CSV = _DATA


@dataclass(frozen=True)
class MixRun:
    """One lineage run's offline handle: its evals_v2 id stem + HF checkpoint steps."""

    mix: str  # W&B `mix=` tag == `_mixture_lineage` Run.mix == results CSV `mix`
    i: int  # sweep index (`i{N}` in the GCS dir / config name)
    gcs_dir: str  # full `dna-bolinas-mix-v0.9-p1B-...` run dir under _GCS_BUCKET
    steps: tuple[int, ...]  # HF-exported steps (ascending; last == final)


# `gcloud storage ls .../hf/` per run, 2026-07-08. mix tag == the run-dir segment
# between `i{N}-` and the trailing hash, and == `_mixture_lineage` Run.mix.
HF_STEPS: tuple[MixRun, ...] = (
    # m5.1 lineage roots + the Fig-9 uniform baseline.
    MixRun("uniform", 0, "dna-bolinas-mix-v0.9-p1B-i0-uniform-2ba217", (10000, 20000, 25004)),
    # Fig-9 uniform→upstream sweep (single final HF step each).
    MixRun("uniform_to_upstream_1", 10, "dna-bolinas-mix-v0.9-p1B-i10-uniform_to_upstream_1-bb0570", (2501,)),
    MixRun("uniform_to_upstream_2", 11, "dna-bolinas-mix-v0.9-p1B-i11-uniform_to_upstream_2-52dd3a", (2501,)),
    MixRun("uniform_to_upstream_3", 12, "dna-bolinas-mix-v0.9-p1B-i12-uniform_to_upstream_3-de18d1", (2500,)),
    MixRun("uniform_to_upstream_4", 13, "dna-bolinas-mix-v0.9-p1B-i13-uniform_to_upstream_4-cdebab", (2499,)),
    MixRun("uniform_to_upstream_5", 14, "dna-bolinas-mix-v0.9-p1B-i14-uniform_to_upstream_5-13a591", (2501,)),
    MixRun("uniform_to_upstream_3.5", 15, "dna-bolinas-mix-v0.9-p1B-i15-uniform_to_upstream_3.5-d0cfb0", (8332,)),
    MixRun("uniform_to_upstream_3.6", 16, "dna-bolinas-mix-v0.9-p1B-i16-uniform_to_upstream_3.6-07bc9c", (8333,)),
    # m5.1 lineage middle stage.
    MixRun("uniform_to_uniform_1", 18, "dna-bolinas-mix-v0.9-p1B-i18-uniform_to_uniform_1-84cd83", (10000, 20000, 29579)),
    # m5.1 lineage leaf.
    MixRun("exp135-zoonomia-m5.1", 24, "dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e", (30000, 40000, 50000, 59158)),
    # m1.3 lineage (zoonomia uniform ⅕): m1 → m1.1 → m1.2 → m1.3.
    MixRun("exp135-zoonomia-m1", 20, "dna-bolinas-mix-v0.9-p1B-i20-exp135-zoonomia-m1-3ea283", (10000, 20000, 29579)),
    MixRun("exp135-zoonomia-m1.1", 26, "dna-bolinas-mix-v0.9-p1B-i26-exp135-zoonomia-m1.1-256ffd", (30000, 40000, 50000, 53243)),
    MixRun("exp135-zoonomia-m1.2", 28, "dna-bolinas-mix-v0.9-p1B-i28-exp135-zoonomia-m1.2-b52d8b", (50000, 60000, 70000, 70991)),
    MixRun("exp135-zoonomia-m1.3", 30, "dna-bolinas-mix-v0.9-p1B-i30-exp135-zoonomia-m1.3-c3a54c", (60000, 70000, 80000, 82823)),
    # m3.3 lineage (zoonomia upstream-tilted): m3 → m3.1 → m3.2 → m3.3.
    MixRun("exp135-zoonomia-m3", 22, "dna-bolinas-mix-v0.9-p1B-i22-exp135-zoonomia-m3-3aae5f", (10000, 20000, 29578)),
    MixRun("exp135-zoonomia-m3.1", 25, "dna-bolinas-mix-v0.9-p1B-i25-exp135-zoonomia-m3.1-e9bc43", (30000, 40000, 50000, 53238)),
    MixRun("exp135-zoonomia-m3.2", 27, "dna-bolinas-mix-v0.9-p1B-i27-exp135-zoonomia-m3.2-da7a05", (50000, 60000, 70000, 70983)),
    MixRun("exp135-zoonomia-m3.3", 29, "dna-bolinas-mix-v0.9-p1B-i29-exp135-zoonomia-m3.3-148abf", (60000, 70000, 80000, 82813)),
)

BY_MIX: dict[str, MixRun] = {r.mix: r for r in HF_STEPS}

# Short forms baked into pre-existing config.yaml entries — reused so we don't
# double-score the same checkpoint under a second model id. Everything else keys
# on the full mix tag (self-describing).
_NAME_SHORT = {
    "uniform_to_upstream_3.6": "upstream",  # config `mix-v0.9-p1B-i16-upstream-step-8333`
    "exp135-zoonomia-m5.1": "exp135-m5.1",  # config `mix-v0.9-p1B-i24-exp135-m5.1-step-59158`
}


def cfg_name(mix: str, step: int) -> str:
    """evals_v2 ``config.yaml`` model id (== S3 results key) for one checkpoint.

    Matches the three pre-existing mixture entries exactly (i0-uniform-25004,
    i16-upstream-8333, i24-exp135-m5.1-59158) so they are reused, not duplicated.
    """
    run = BY_MIX[mix]
    short = _NAME_SHORT.get(mix, mix)
    return f"mix-v0.9-p1B-i{run.i}-{short}-step-{step}"


def gcs_path(mix: str, step: int) -> str:
    """Full ``gs://.../hf/step-N`` path for one checkpoint (config ``gcs_path``)."""
    return f"{_GCS_BUCKET}/{BY_MIX[mix].gcs_dir}/hf/step-{step}"


def final_name(mix: str) -> str:
    """Config id of a run's final (highest) HF checkpoint — the Fig-9 datapoint."""
    return cfg_name(mix, BY_MIX[mix].steps[-1])


def config_entries() -> Iterator[dict[str, object]]:
    """Every mixture checkpoint (Fig 9 finals ∪ Fig 10 lineage steps) as an
    evals_v2 config entry dict, scored on BOTH mendelian_traits and sge."""
    for run in HF_STEPS:
        for step in run.steps:
            yield {
                "name": cfg_name(run.mix, step),
                "gcs_path": gcs_path(run.mix, step),
                "window_size": _WINDOW_SIZE,
                "datasets": ["mendelian_traits", "sge"],
            }


@lru_cache(maxsize=1)
def results_frame() -> pd.DataFrame:
    """Eric's run-metadata table indexed by ``mix`` (tokens / num_train_steps /
    state). Training-intrinsic — reused verbatim for the token-axis mapping."""
    return pd.read_csv(RESULTS_CSV).set_index("mix")


def own_tokens() -> dict[str, float]:
    """``mix -> own new-portion token budget`` (the W&B ``tokens`` tag), the input
    ``_mixture_lineage.inherited_components`` / ``cumulative_total`` expect."""
    res = results_frame()
    return {r.mix: float(res.loc[r.mix, "tokens"]) for r in HF_STEPS if r.mix in res.index}
