"""Shared data prep for the parameter-scaling downstream figures (Figs 5–6).

Reads the 8 scaling-ladder **endpoints'** new-eval AUPRC (via
``blog_metrics.read_llr_metrics``) joined with ``params`` + validation ``eval_loss``
from the vendored ``parameter_scaling_results.csv`` (training results are
eval-independent). Figs 7/8 (training curves / loss↔AUPRC correlation over
training) read the ladder *intermediates* scored by #364 the same way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from marin_dna.pipelines.evals.blog_metrics import read_llr_metrics, read_probe_metrics

DATA = Path(__file__).resolve().parent / "data" / "parameter_scaling_results.csv"
LADDER_FINAL_STEP = 215573

# Subset → display label, in EARTH_QUAL color-slot order (Eric's VEP_PANELS), so a
# variant type keeps its color across Figs 5/6.
VEP_PANELS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("tss_proximal", "promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("splicing", "splicing"),
    ("synonymous_variant", "synonymous"),
)


def ladder_table(read_fn) -> pd.DataFrame:
    """Long-form ``(subset, value, n, params, eval_loss)`` for the 8 ladder endpoints.

    ``read_fn`` selects the world — ``read_llr_metrics`` (M·LLR) or
    ``read_probe_metrics`` (M·Probe). ``value`` = per-subset AUPRC; ``params`` /
    ``eval_loss`` come from the vendored scaling CSV (joined by run-name stem).
    """
    meta = pd.read_csv(DATA)[["run_name", "params", "eval_loss"]]
    frames = []
    for _, r in meta.iterrows():
        stem = r["run_name"].removeprefix("dna-bolinas-")  # scaling-v0.5-hH-pP
        df = read_fn(f"{stem}-step-{LADDER_FINAL_STEP}", "mendelian_traits").to_pandas()
        df["params"] = int(r["params"])
        df["eval_loss"] = float(r["eval_loss"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ladder_llr_table() -> pd.DataFrame:
    """M·LLR convenience wrapper for :func:`ladder_table`."""
    return ladder_table(read_llr_metrics)


def ladder_probe_table() -> pd.DataFrame:
    """M·Probe convenience wrapper for :func:`ladder_table`."""
    return ladder_table(read_probe_metrics)
