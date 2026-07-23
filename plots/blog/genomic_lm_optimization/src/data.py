"""Fetch wandb data for the Bolinas DNA transfer-validation and parameter-scaling sweeps.

Writes two CSVs under `data/`:
  - data/transfer_validation_results.csv  (v0.14 + v0.15)
  - data/parameter_scaling_results.csv    (v0.5, exactly 8 models)

Each row corresponds to one wandb run. Hparams come from `run.config`; `params` and
`tokens` come from the run's tag list (`params=...`, `tokens=...`); `eval/loss` and
the lm_eval AUPRC metrics come from `run.summary` (final logged value).

A run is considered "complete" — and thus included — if either:
  - `state == "finished"`, or
  - `state == "crashed"` AND `run_progress > 0.99`
Some runs failed during artifact upload at the very end of training but successfully
logged their final eval metrics; the second branch keeps those.

Usage:
    uv run src/data.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

WANDB_PROJECT = "eric-czech/marin"

TRANSFER_VERSIONS = ("v0.14", "v0.15")
SCALING_VERSION = "v0.5"
SCALING_EXPECTED_MODELS = 8

MIXTURE_VERSION = "v0.9"
# Warm-started from an external (param-scaling) run rather than anything in this
# sweep, so it has no in-sweep lineage — excluded from the mixture tree/CSV.
MIXTURE_EXCLUDE: frozenset[str] = frozenset({"cont_ps_up_down_1"})
# Non-finished runs kept as an exception: cds_only crashed at ~70% but logged its
# final-eligible evals, and it's the only C-only baseline. Its token count is
# scaled by run_progress downstream (see figures/appendix/mixture_tree.py) since
# it never completed training.
#
# Zoonomia continuation chains (trees 5 and 6 in the lineage) ran on preemptible
# VMs and could temporarily show as `crashed`/`running` between resumes. Keep the
# exception list so the trajectory figures remain robust if these runs are
# refreshed during a transient state; once finished, the normal finished-run path
# applies.
MIXTURE_INFLIGHT: frozenset[str] = frozenset(
    {
        "exp135-zoonomia-m1.1",
        "exp135-zoonomia-m1.2",
        "exp135-zoonomia-m1.3",
        "exp135-zoonomia-m3.1",
        "exp135-zoonomia-m3.2",
        "exp135-zoonomia-m3.3",
    }
)
MIXTURE_INCLUDE_UNFINISHED: frozenset[str] = frozenset({"cds_only"}) | MIXTURE_INFLIGHT

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# lm_eval keys logged flat (slash-separated) in run.summary.
LM_EVAL_KEYS: tuple[str, ...] = (
    "lm_eval/traitgym_mendelian_v2_255/auprc",
    "lm_eval/traitgym_mendelian_v2_255/3_prime_UTR_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/5_prime_UTR_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/distal/auprc",
    "lm_eval/traitgym_mendelian_v2_255/missense_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/non_coding_transcript_exon_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/splicing/auprc",
    "lm_eval/traitgym_mendelian_v2_255/synonymous_variant/auprc",
    "lm_eval/traitgym_mendelian_v2_255/tss_proximal/auprc",
)

TRANSFER_AXES = ("learning_rate", "beta2", "epsilon")
TRANSFER_CONTROL_ROLES = ("positive-control", "negative-control")

# Hparam columns and the candidate paths to look them up at in the flattened
# run.config. Levanter logs nested dataclasses with dot-separated keys; we try
# each candidate in order and fail loudly if none match.
HPARAM_LOOKUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hidden_size", ("model.hidden_dim",)),
    ("batch_size", ("trainer.train_batch_size",)),
    ("num_train_steps", ("trainer.num_train_steps",)),
    ("train_seq_len", ("train_seq_len",)),
    ("learning_rate", ("optimizer.learning_rate",)),
    ("adam_lr", ("optimizer.adam_lr",)),
    ("beta1", ("optimizer.beta1",)),
    ("beta2", ("optimizer.beta2",)),
    ("epsilon", ("optimizer.epsilon",)),
    ("max_grad_norm", ("optimizer.max_grad_norm",)),
    ("z_loss_weight", ("z_loss_weight",)),
    ("initializer_range", ("model.initializer_range",)),
)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dicts with dot-separated keys."""
    out: dict = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full))
        else:
            out[full] = v
    return out


def _lookup(flat: dict, candidates: tuple[str, ...], run_name: str):
    for c in candidates:
        if c in flat:
            return flat[c]
    raise KeyError(
        f"run {run_name!r}: none of {candidates} found in flattened config "
        f"(sample keys: {sorted(flat)[:30]!r})"
    )


def _tag_int(tags: list[str], key: str, run_name: str) -> int:
    prefix = f"{key}="
    matches = [t[len(prefix) :] for t in tags if t.startswith(prefix)]
    if not matches:
        raise KeyError(f"run {run_name!r}: tag {prefix}... not found in {tags}")
    if len(matches) > 1:
        raise ValueError(f"run {run_name!r}: multiple {prefix}... tags: {matches}")
    return int(matches[0])


def _summary_float(summary, key: str) -> float:
    v = summary.get(key)
    if v is None:
        return float("nan")
    f = float(v)
    return f if np.isfinite(f) else float("nan")


# Levanter logs cumulative compute as `throughput/total_gflops`. We expose it as
# `tflops` (rounded int) — keeps the column compact and avoids float overflow.
THROUGHPUT_GFLOPS_KEY = "throughput/total_gflops"


def _tflops_from_gflops(v):
    """Convert a `throughput/total_gflops` value to TFLOPs (rounded int).

    Returns pd.NA when missing/non-finite — runs that crashed before logging a
    single step (state=failed, _step=None) have no throughput recorded.
    """
    if v is None:
        return pd.NA
    f = float(v)
    if not np.isfinite(f):
        return pd.NA
    return int(round(f / 1000.0))


def _parse_transfer_role(run_name: str) -> str:
    for role in TRANSFER_CONTROL_ROLES:
        if run_name.endswith(f"-{role}"):
            return role
    for axis in TRANSFER_AXES:
        if re.search(rf"-{re.escape(axis)}-\d+$", run_name):
            return axis
    raise ValueError(f"could not parse transfer role from run name: {run_name!r}")


def _base_row(run, version: str) -> dict:
    flat = _flatten(dict(run.config))
    row = {
        "run_name": run.name,
        "version": version,
        "state": run.state,
        "params": _tag_int(run.tags, "params", run.name),
        "tokens": _tag_int(run.tags, "tokens", run.name),
    }
    for col, candidates in HPARAM_LOOKUPS:
        row[col] = _lookup(flat, candidates, run.name)
    # Fraction of training completed: latest logged step / configured total steps.
    last_step = run.summary.get("_step")
    total_steps = row["num_train_steps"]
    row["run_progress"] = (
        float(last_step) / total_steps
        if last_step is not None and total_steps
        else float("nan")
    )
    row["tflops"] = _tflops_from_gflops(run.summary.get(THROUGHPUT_GFLOPS_KEY))
    row["eval_loss"] = _summary_float(run.summary, "eval/loss")
    for region in ("cds", "upstream", "downstream"):
        row[f"eval_loss_{region}"] = _summary_float(
            run.summary, f"eval/val_{region}/loss"
        )
    return row


CRASHED_PROGRESS_THRESHOLD = 0.99


def _is_complete(run) -> bool:
    """A run is complete if finished, or crashed but having logged ~all training steps.

    The second case covers runs that crashed during end-of-training artifact upload
    after their final eval metrics had already been logged — see module docstring.
    """
    if run.state == "finished":
        return True
    if run.state != "crashed":
        return False
    last_step = run.summary.get("_step")
    if last_step is None:
        return False
    flat = _flatten(dict(run.config))
    total_steps = _lookup(flat, ("trainer.num_train_steps",), run.name)
    if not total_steps:
        return False
    return float(last_step) / float(total_steps) > CRASHED_PROGRESS_THRESHOLD


def _fetch_runs(api, name_prefix: str) -> list:
    runs = list(
        api.runs(
            WANDB_PROJECT,
            filters={"display_name": {"$regex": f"^{re.escape(name_prefix)}"}},
        )
    )
    complete = [r for r in runs if _is_complete(r)]
    dropped = len(runs) - len(complete)
    msg = f"  found {len(runs)} runs matching {name_prefix!r}"
    if dropped:
        msg += f" (kept {len(complete)} complete, dropped {dropped} incomplete)"
    print(msg)
    return complete


def fetch_transfer_validation(api) -> pd.DataFrame:
    print(f"Fetching transfer-validation runs from {WANDB_PROJECT} ...")
    rows: list[dict] = []
    for version in TRANSFER_VERSIONS:
        # Match all version-prefixed runs, then drop checkpoint VEP eval runs
        # (they share the prefix but aren't training runs).
        prefix = f"dna-bolinas-transfer-{version}-"
        for run in _fetch_runs(api, prefix):
            if "checkpoint_vep_eval" in run.tags:
                continue
            row = _base_row(run, version)
            row["role"] = _parse_transfer_role(run.name)
            rows.append(row)
    df = pd.DataFrame.from_records(rows)
    df["tflops"] = df["tflops"].astype("Int64")
    cols = [
        "run_name",
        "version",
        "state",
        "run_progress",
        "role",
        "hidden_size",
        "params",
        "tokens",
        "tflops",
        "batch_size",
        "num_train_steps",
        "train_seq_len",
        "learning_rate",
        "adam_lr",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "z_loss_weight",
        "initializer_range",
        "eval_loss",
        "eval_loss_cds",
        "eval_loss_upstream",
        "eval_loss_downstream",
    ]
    df = df[cols].sort_values(["version", "params", "run_name"]).reset_index(drop=True)
    return df


def fetch_parameter_scaling(api) -> pd.DataFrame:
    print(f"Fetching parameter-scaling runs from {WANDB_PROJECT} ...")
    prefix = f"dna-bolinas-scaling-{SCALING_VERSION}-"
    runs = _fetch_runs(api, prefix)
    rows: list[dict] = []
    for run in runs:
        row = _base_row(run, SCALING_VERSION)
        for key in LM_EVAL_KEYS:
            row[key] = _summary_float(run.summary, key)
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    df["tflops"] = df["tflops"].astype("Int64")
    cols = [
        "run_name",
        "version",
        "state",
        "run_progress",
        "hidden_size",
        "params",
        "tokens",
        "tflops",
        "batch_size",
        "num_train_steps",
        "train_seq_len",
        "learning_rate",
        "adam_lr",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "z_loss_weight",
        "initializer_range",
        "eval_loss",
        "eval_loss_cds",
        "eval_loss_upstream",
        "eval_loss_downstream",
        *LM_EVAL_KEYS,
    ]
    df = df[cols].sort_values(["params", "run_name"]).reset_index(drop=True)
    if len(df) != SCALING_EXPECTED_MODELS:
        raise AssertionError(
            f"Expected exactly {SCALING_EXPECTED_MODELS} scaling-sweep runs for "
            f"{SCALING_VERSION}, found {len(df)}: {df['run_name'].tolist()}"
        )
    return df


SCALING_HISTORY_METRICS: tuple[str, ...] = ("train/loss", "eval/loss")


def fetch_parameter_scaling_history(api) -> pd.DataFrame:
    """Pull train/loss and eval/loss history for each scaling-sweep run, downsampled
    to the (sparse) eval steps so train/loss is co-sampled with eval/loss.

    Long form: one row per (run_name, step, metric). `train/loss` is logged every
    step but we keep only the steps where `eval/loss` is also logged — wandb's
    `scan_history(keys=...)` filters to records where every requested key is
    present, so a single scan with all three keys returns the eval-step rows
    with all values populated.
    """
    print(f"Fetching parameter-scaling history from {WANDB_PROJECT} ...")
    prefix = f"dna-bolinas-scaling-{SCALING_VERSION}-"
    runs = _fetch_runs(api, prefix)
    if len(runs) != SCALING_EXPECTED_MODELS:
        raise AssertionError(
            f"Expected exactly {SCALING_EXPECTED_MODELS} scaling-sweep runs for "
            f"{SCALING_VERSION}, found {len(runs)}: {[r.name for r in runs]}"
        )
    frames: list[pd.DataFrame] = []
    for run in runs:
        print(f"  fetching history for {run.name} ...")
        # `run.history(keys=...)` does server-side AND-filtering in a single
        # GraphQL call — since eval/loss is sparse, the train/loss × eval/loss
        # intersection is the ~34 eval-step rows, returned in seconds.
        # `scan_history` paginates the full 215k-step underlying sequence and
        # is ~100x slower for this query.
        df = run.history(keys=list(SCALING_HISTORY_METRICS), samples=10000, pandas=True)
        if df.empty:
            print("    0 eval steps")
            continue
        df = df[["_step", *SCALING_HISTORY_METRICS]].rename(columns={"_step": "step"})
        long = df.melt(
            id_vars=["step"],
            value_vars=list(SCALING_HISTORY_METRICS),
            var_name="metric",
            value_name="value",
        )
        long["run_name"] = run.name
        frames.append(long)
        print(
            f"    {len(df)} eval steps × {len(SCALING_HISTORY_METRICS)} metrics = {len(long)} rows"
        )
    out = pd.concat(frames, ignore_index=True)
    out["step"] = out["step"].astype(int)
    out = out[["run_name", "metric", "step", "value"]]
    return out.sort_values(["run_name", "metric", "step"]).reset_index(drop=True)


def _tag_str(tags: list[str], key: str, run_name: str) -> str:
    prefix = f"{key}="
    matches = [t[len(prefix) :] for t in tags if t.startswith(prefix)]
    if not matches:
        raise KeyError(f"run {run_name!r}: tag {prefix}... not found in {tags}")
    if len(matches) > 1:
        raise ValueError(f"run {run_name!r}: multiple {prefix}... tags: {matches}")
    return matches[0]


def _mixture_runs(api) -> list[tuple]:
    """Included v0.9 mixture runs as (run, mix) pairs: finished (plus the
    cds_only exception in MIXTURE_INCLUDE_UNFINISHED), minus MIXTURE_EXCLUDE.
    """
    prefix = f"dna-bolinas-mix-{MIXTURE_VERSION}-"
    runs = list(
        api.runs(
            WANDB_PROJECT, filters={"display_name": {"$regex": f"^{re.escape(prefix)}"}}
        )
    )
    out: list[tuple] = []
    for run in runs:
        mix = _tag_str(run.tags, "mix", run.name)
        if mix in MIXTURE_EXCLUDE:
            continue
        if run.state != "finished" and mix not in MIXTURE_INCLUDE_UNFINISHED:
            continue
        out.append((run, mix))
    print(f"  kept {len(out)} runs (matched {len(runs)})")
    return out


def fetch_mixture(api) -> pd.DataFrame:
    """Final-only rows for the v0.9 mixture sweep (one row per included run).

    `params`/`tokens` come from tags (`tokens` is the run's *own* new-portion token
    count); the lineage / mixture weights / cumulative token accounting live in
    `src/figures/mixture_lineage.py`, keyed by the `mix` column here. All metrics
    here are final (run.summary); per-eval trajectories (used for begin→end deltas)
    live in `data_mixture_history.csv` via `fetch_mixture_history`.
    """
    print(f"Fetching mixture-sweep runs from {WANDB_PROJECT} ...")
    rows: list[dict] = []
    for run, mix in _mixture_runs(api):
        row = _base_row(run, MIXTURE_VERSION)
        row["mix"] = mix
        for key in LM_EVAL_KEYS:
            row[key] = _summary_float(run.summary, key)
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    df["tflops"] = df["tflops"].astype("Int64")
    cols = [
        "run_name",
        "mix",
        "version",
        "state",
        "run_progress",
        "hidden_size",
        "params",
        "tokens",
        "tflops",
        "batch_size",
        "num_train_steps",
        "train_seq_len",
        "learning_rate",
        "adam_lr",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "z_loss_weight",
        "initializer_range",
        "eval_loss",
        "eval_loss_cds",
        "eval_loss_upstream",
        "eval_loss_downstream",
        *LM_EVAL_KEYS,
    ]
    df = df[cols].sort_values("mix").reset_index(drop=True)
    return df


def fetch_mixture_history(api) -> pd.DataFrame:
    """Per-eval TraitGym AUPRC trajectories for the mixture runs (long form).

    One row per (run_name, mix, metric, step). Used for begin→end deltas (e.g.
    Figure 7): the first logged eval lands at ~num_train_steps/10, the last at the
    end of training. `run.history(keys=...)` AND-filters to the (sparse) eval steps
    where all requested keys are present.
    """
    print(f"Fetching mixture-sweep history from {WANDB_PROJECT} ...")
    frames: list[pd.DataFrame] = []
    for run, mix in _mixture_runs(api):
        df = run.history(keys=list(LM_EVAL_KEYS), samples=10000, pandas=True)
        present = [k for k in LM_EVAL_KEYS if k in df.columns]
        if df.empty or "_step" not in df.columns or not present:
            print(f"  {mix}: no eval history")
            continue
        df = df[["_step", *present]].rename(columns={"_step": "step"})
        long = df.melt(
            id_vars=["step"], value_vars=present, var_name="metric", value_name="value"
        ).dropna(subset=["value"])
        long["run_name"] = run.name
        long["mix"] = mix
        frames.append(long)
        print(f"  {mix}: {df['step'].nunique()} eval steps")
    out = pd.concat(frames, ignore_index=True)
    out["step"] = out["step"].astype(int)
    out = out[["run_name", "mix", "metric", "step", "value"]]
    return out.sort_values(["mix", "metric", "step"]).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=300)

    transfer_df = fetch_transfer_validation(api)
    transfer_path = DATA_DIR / "transfer_validation_results.csv"
    transfer_df.to_csv(transfer_path, index=False)
    print(f"Wrote {len(transfer_df)} rows to {transfer_path}")

    scaling_df = fetch_parameter_scaling(api)
    scaling_path = DATA_DIR / "parameter_scaling_results.csv"
    scaling_df.to_csv(scaling_path, index=False)
    print(f"Wrote {len(scaling_df)} rows to {scaling_path}")

    history_df = fetch_parameter_scaling_history(api)
    history_path = DATA_DIR / "parameter_scaling_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Wrote {len(history_df)} rows to {history_path}")

    mixture_df = fetch_mixture(api)
    mixture_path = DATA_DIR / "data_mixture_results.csv"
    mixture_df.to_csv(mixture_path, index=False)
    print(f"Wrote {len(mixture_df)} rows to {mixture_path}")

    mixture_history_df = fetch_mixture_history(api)
    mixture_history_path = DATA_DIR / "data_mixture_history.csv"
    mixture_history_df.to_csv(mixture_history_path, index=False)
    print(f"Wrote {len(mixture_history_df)} rows to {mixture_history_path}")


if __name__ == "__main__":
    sys.exit(main())
