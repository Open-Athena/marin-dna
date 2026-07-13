"""Shared data prep for the parameter-scaling downstream figures (Figs 5–6).

Reads the 8 scaling-ladder **endpoints'** new-eval AUPRC (via
``blog_metrics.read_llr_metrics``) joined with ``params`` + the validation loss
for the training-data region corresponding to each variant type from the
vendored ``parameter_scaling_results.csv``. Figs 7/8 (training curves /
loss↔AUPRC correlation over training) read the ladder *intermediates* scored by
#364 separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from marin_dna.pipelines.evals.blog_metrics import read_llr_metrics, read_probe_metrics
from plots.blog._regions import VARIANT_REGION

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

REGION_LOSS_COLUMNS: tuple[str, ...] = tuple(
    f"eval_loss_{region}" for region in dict.fromkeys(VARIANT_REGION.values())
)


def add_relevant_region_loss(
    data: pd.DataFrame, metadata_row: pd.Series
) -> pd.DataFrame:
    """Attach the corresponding-region validation loss to each variant row.

    ``eval_loss`` in the source CSV is the global validation loss. Fig 6 instead
    uses ``eval_loss_{region}``, selected via :data:`VARIANT_REGION`, so e.g.
    missense is paired with CDS loss and promoter with upstream loss. Assertions
    make unsupported variants or missing/non-finite loss values fail loudly.
    """
    out = data.copy()
    out["loss_region"] = out["subset"].map(VARIANT_REGION)
    assert out["loss_region"].notna().all(), (
        "missing variant→region mapping for "
        f"{sorted(out.loc[out['loss_region'].isna(), 'subset'].unique())}"
    )
    loss_by_region = {
        region: float(metadata_row[f"eval_loss_{region}"])
        for region in out["loss_region"].unique()
    }
    assert all(np.isfinite(loss) for loss in loss_by_region.values()), (
        f"non-finite region loss(es): {loss_by_region}"
    )
    out["eval_loss"] = out["loss_region"].map(loss_by_region)
    assert np.isfinite(out["eval_loss"]).all()
    return out


def ladder_table(read_fn) -> pd.DataFrame:
    """Long-form endpoint metrics with variant-matched regional validation loss.

    ``read_fn`` selects the world — ``read_llr_metrics`` (M·LLR) or
    ``read_probe_metrics`` (M·Probe). ``value`` = per-subset AUPRC; ``params`` /
    ``eval_loss`` come from the vendored scaling CSV (joined by run-name stem),
    with loss selected per variant using :data:`VARIANT_REGION`.
    """
    meta = pd.read_csv(DATA)[["run_name", "params", *REGION_LOSS_COLUMNS]]
    expected_subsets = {subset for subset, _ in VEP_PANELS}
    frames = []
    for _, r in meta.iterrows():
        stem = r["run_name"].removeprefix("dna-bolinas-")  # scaling-v0.5-hH-pP
        df = read_fn(f"{stem}-step-{LADDER_FINAL_STEP}", "mendelian_traits").to_pandas()
        df = df[df["subset"].isin(expected_subsets)].copy()
        assert set(df["subset"]) == expected_subsets, (
            f"{stem}: expected Mendelian subsets {sorted(expected_subsets)}, "
            f"got {sorted(df['subset'].unique())}"
        )
        df["params"] = int(r["params"])
        frames.append(add_relevant_region_loss(df, r))
    return pd.concat(frames, ignore_index=True)


def ladder_llr_table() -> pd.DataFrame:
    """M·LLR convenience wrapper for :func:`ladder_table`."""
    return ladder_table(read_llr_metrics)


def ladder_probe_table() -> pd.DataFrame:
    """M·Probe convenience wrapper for :func:`ladder_table`."""
    return ladder_table(read_probe_metrics)
