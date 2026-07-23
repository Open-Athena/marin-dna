"""Data wiring for the figure set: input/output paths, CSV loaders, and the
shared `save` helper. Loaders return cleaned DataFrames ready to plot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.savefig import save_figure

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "transfer_validation_results.csv"
SCALING_RESULTS_PATH = ROOT / "data" / "parameter_scaling_results.csv"
SCALING_HISTORY_PATH = ROOT / "data" / "parameter_scaling_history.csv"
MIXTURE_RESULTS_PATH = ROOT / "data" / "data_mixture_results.csv"
MIXTURE_HISTORY_PATH = ROOT / "data" / "data_mixture_history.csv"
MODEL_LEADERBOARD_PATH = ROOT / "data" / "model_leaderboard.csv"
FIGURES_DIR = ROOT.parents[2] / "plots" / "output" / "blog" / "genomic_lm_optimization"

# Eval VEP sample sizes per variant type (from docs/outline.md). Embedded in
# Figure 6's panel titles. Figure 5 reuses the same ordering for consistency.
VEP_PANELS: tuple[tuple[str, str, int], ...] = (
    ("missense_variant", "missense", 14800),
    ("tss_proximal", "promoter", 1800),
    ("5_prime_UTR_variant", "5' UTR", 2100),
    ("3_prime_UTR_variant", "3' UTR", 770),
    ("splicing", "splicing", 2670),
    ("synonymous_variant", "synonymous", 460),
)


def _round_sig(x: float, sig: int = 6) -> float:
    """Round `x` to `sig` significant figures (NaN-safe). Snaps float-precision-jittered values."""
    if x is None or pd.isna(x) or x == 0:
        return x
    return float(f"{x:.{sig}g}")


def load_transfer() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Different runs can store the same nominal sweep value with slightly different
    # float bit patterns (e.g. 2e-9 vs 1.999999999999999e-9), which then show up as
    # duplicate x-axis ticks. Snap sweep-axis fields to 6 sig figs to fix this.
    for col in ("learning_rate", "beta2", "epsilon"):
        df[col] = df[col].apply(_round_sig)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df


def load_scaling_results() -> pd.DataFrame:
    df = pd.read_csv(SCALING_RESULTS_PATH)
    print(f"Loaded {len(df)} rows from {SCALING_RESULTS_PATH}")
    return df


def load_scaling_history() -> pd.DataFrame:
    df = pd.read_csv(SCALING_HISTORY_PATH)
    print(f"Loaded {len(df)} rows from {SCALING_HISTORY_PATH}")
    return df


def load_mixture() -> pd.DataFrame:
    df = pd.read_csv(MIXTURE_RESULTS_PATH)
    print(f"Loaded {len(df)} rows from {MIXTURE_RESULTS_PATH}")
    return df


def load_mixture_history() -> pd.DataFrame:
    df = pd.read_csv(MIXTURE_HISTORY_PATH)
    print(f"Loaded {len(df)} rows from {MIXTURE_HISTORY_PATH}")
    return df


def load_leaderboard() -> pd.DataFrame:
    # First line is a provenance comment; the header is on line 2.
    df = pd.read_csv(MODEL_LEADERBOARD_PATH, skiprows=1)
    print(f"Loaded {len(df)} rows from {MODEL_LEADERBOARD_PATH}")
    return df


def save(fig, name: str) -> None:
    save_figure(fig, FIGURES_DIR, name)
