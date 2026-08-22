"""Paired within-run Mendelian VEP analysis for exp479."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp479_mntp.vep_metrics import paired_ap_delta


def paired_mendelian_trajectory(
    scores: pd.DataFrame,
    *,
    n_bootstrap: int = 2_000,
    baseline_step: int = 0,
) -> pd.DataFrame:
    """Compare each Mendelian macro checkpoint with the paired step-0 rows."""

    required = {"label", "subset", "match_group"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Mendelian score frame lacks columns {sorted(missing)}")
    score_columns = {
        int(column.removeprefix("step_")): column
        for column in scores.columns
        if column.startswith("step_")
    }
    if baseline_step not in score_columns:
        raise ValueError(f"missing baseline step {baseline_step}")
    baseline = score_columns[baseline_step]
    qualifying = [
        cell
        for _, cell in scores.groupby("subset", sort=False)
        if cell["match_group"].nunique() >= 30
    ]
    if not qualifying:
        raise ValueError("no Mendelian subsets have at least 30 match groups")

    rows: list[dict[str, float | int]] = []
    for step, candidate in sorted(score_columns.items()):
        if step == baseline_step:
            rows.append(
                {
                    "optimizer_step": step,
                    "delta": 0.0,
                    "se": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n_subsets": len(qualifying),
                    "n_rows": sum(len(cell) for cell in qualifying),
                }
            )
            continue
        children = [
            paired_ap_delta(
                cell["label"],
                cell[candidate],
                cell[baseline],
                cell["match_group"],
                n_bootstrap=n_bootstrap,
                seed=0,
            )
            for cell in qualifying
        ]
        count = len(children)
        delta = float(sum(float(row["delta"]) for row in children) / count)
        se = float(math.sqrt(sum(float(row["se"]) ** 2 for row in children)) / count)
        rows.append(
            {
                "optimizer_step": step,
                "delta": delta,
                "se": se,
                "ci_low": delta - 1.96 * se,
                "ci_high": delta + 1.96 * se,
                "n_subsets": count,
                "n_rows": sum(int(row["n_rows"]) for row in children),
            }
        )
    return pd.DataFrame(rows)


def plot_mendelian_trajectory(
    endpoints: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot absolute Mendelian AUPRC and paired changes from step 0."""

    mendelian = endpoints[endpoints["dataset"] == "mendelian_traits"].copy()
    merged = mendelian.merge(comparisons, on="optimizer_step", validate="one_to_one")
    merged = merged.sort_values("optimizer_step")
    if len(merged) != len(comparisons):
        raise ValueError("Mendelian endpoints and paired comparisons have different steps")
    baseline = float(merged.loc[merged["optimizer_step"] == 0, "auprc"].item())
    if not np.allclose(merged["auprc"] - baseline, merged["delta"], atol=1e-12):
        raise ValueError("paired Mendelian deltas disagree with absolute endpoints")

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    color = "#4C78A8"
    steps = merged["optimizer_step"].to_numpy()
    axes[0].plot(steps, merged["auprc"], color=color, marker="o", linewidth=2)
    axes[0].axhline(baseline, color="#555555", linestyle="--", linewidth=1.4)
    candidate = merged[merged["optimizer_step"] != 0]
    axes[0].errorbar(
        candidate["optimizer_step"],
        candidate["auprc"],
        yerr=np.vstack(
            [
                candidate["delta"] - candidate["ci_low"],
                candidate["ci_high"] - candidate["delta"],
            ]
        ),
        fmt="none",
        ecolor=color,
        alpha=0.55,
        capsize=3,
    )
    axes[0].set_ylabel("Mendelian macro AUPRC")
    axes[0].set_title("Mendelian VEP shows no resolved within-run change")

    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=1.4)
    axes[1].errorbar(
        candidate["optimizer_step"],
        candidate["delta"],
        yerr=np.vstack(
            [
                candidate["delta"] - candidate["ci_low"],
                candidate["ci_high"] - candidate["delta"],
            ]
        ),
        fmt="o-",
        color=color,
        linewidth=2,
        capsize=3,
    )
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Paired AUPRC change\nversus step 0")

    for axis in axes:
        axis.axvline(100, color="#BBBBBB", linestyle=":", linewidth=1)
        axis.axvline(800, color="#BBBBBB", linestyle=":", linewidth=1)
        axis.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
