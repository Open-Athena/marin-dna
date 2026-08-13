"""Aggregate-only distal trajectories for issue #459.

exp326 and exp351 did not publish compatible per-variant score parquets, so
their distal patch is deliberately limited to the AUPRC points logged by the
training harness. No soft metric is inferred from these aggregate histories.
"""

import argparse
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from marin_dna_evals.soft_vep_analysis import DATASET, S3_SCORE_ROOT, SPLIT

DISTAL_KEY = "lm_eval/mendelian_traits_255/distal/avg/auprc"
EXP232_CCRE_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 4000, 4500, 4999)
WANDB_RUNS = {
    "exp326 A: no exon overlap": (
        "exp326",
        "gonzalobenegas/marin/"
        "dna-exp326-zoonomia-v1-0p25b-v4_ccre_noexon-v0.1-332e6d",
    ),
    "exp326 B: no exon overlap, enhancer-only": (
        "exp326",
        "gonzalobenegas/marin/"
        "dna-exp326-zoonomia-v1-0p25b-v4_ccre_noexon_enhancer-v0.1-9382fe",
    ),
    "exp351 tiled": (
        "exp351",
        "gonzalobenegas/marin/"
        "dna-exp351-zoonomia-v1-0p25b-tiled-v0.1-812cad",
    ),
    "exp351 centered": (
        "exp351",
        "gonzalobenegas/marin/"
        "dna-exp351-zoonomia-v1-0p25b-centered-v0.1-8adcec",
    ),
}
EXPECTED_FINAL = {
    "exp232 cCRE baseline": 0.127,
    "exp326 A: no exon overlap": 0.299,
    "exp326 B: no exon overlap, enhancer-only": 0.272,
    "exp351 tiled": 0.308,
    "exp351 centered": 0.366,
}
COLORS = {
    "exp232 cCRE baseline": "#7f7f7f",
    "exp326 A: no exon overlap": "#0072B2",
    "exp326 B: no exon overlap, enhancer-only": "#D55E00",
    "exp351 tiled": "#0072B2",
    "exp351 centered": "#D55E00",
}


def exp232_ccre_metric_uri(step: int) -> str:
    """Stored exp232 cCRE metric parquet for one real checkpoint."""
    assert step in EXP232_CCRE_STEPS
    score_uri = (
        f"{S3_SCORE_ROOT}/exp232-v4_ccre_non_promoter-step-{step}/"
        f"{DATASET}.parquet"
    )
    return score_uri.replace("/results/scores/", "/results/metrics/")


def history_rows(
    history: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    experiment: str,
    run_path: str,
) -> pd.DataFrame:
    """Preserve every finite logged distal point in scan order."""
    rows = []
    for history_index, row in enumerate(history):
        step = row.get("_step")
        value = row.get(DISTAL_KEY)
        if step is None or value is None or pd.isna(value):
            continue
        rows.append(
            {
                "experiment": experiment,
                "arm": arm,
                "step": int(step),
                "history_index": history_index,
                "auprc": float(value),
                "source_protocol": "online_lm_eval_fwd_rc_avg",
                "source": f"https://wandb.ai/{run_path}",
            }
        )
    result = pd.DataFrame(rows)
    assert not result.empty, f"no finite {DISTAL_KEY!r} points found for {run_path}"
    return result


def read_exp232_ccre_trajectory() -> pd.DataFrame:
    """Read the stored offline evals_v2 cCRE distal AUPRC checkpoints."""
    rows = []
    for history_index, step in enumerate(EXP232_CCRE_STEPS):
        metric = pl.read_parquet(
            exp232_ccre_metric_uri(step),
            columns=["score_type", "subset", "value", "split"],
            storage_options={"aws_region": "us-east-2"},
        ).filter(
            (pl.col("score_type") == "minus_llr_avg")
            & (pl.col("subset") == "distal")
            & (pl.col("split") == SPLIT)
        )
        assert len(metric) == 1, f"expected one stored distal row at step {step}"
        rows.append(
            {
                "experiment": "exp326",
                "arm": "exp232 cCRE baseline",
                "step": step,
                "history_index": history_index,
                "auprc": float(metric["value"][0]),
                "source_protocol": "offline_evals_v2_minus_llr_avg",
                "source": exp232_ccre_metric_uri(step),
            }
        )
    return pd.DataFrame(rows)


def plot_distal_aggregate(trajectory: pd.DataFrame, output_path: Path) -> None:
    """Render experiment-local distal comparisons without implying soft scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, experiment in zip(axes, ("exp326", "exp351")):
        data = trajectory[trajectory["experiment"] == experiment]
        for arm, arm_data in data.groupby("arm", sort=False):
            arm_data = arm_data.sort_values(["step", "history_index"])
            axis.plot(
                arm_data["step"],
                arm_data["auprc"],
                color=COLORS[arm],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=arm,
            )
        axis.axhline(0.1, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        axis.set_title(experiment)
        axis.set_xlabel("Logged training step")
        axis.grid(alpha=0.25, linewidth=0.7)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Distal Mendelian AUPRC")
    fig.suptitle(
        "Aggregate-only distal trajectories (no compatible per-variant scores)\n"
        "exp232 baseline is offline evals_v2; exp326/351 points are online lm_eval",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def run_distal_patch(output_dir: Path) -> dict[str, Path]:
    """Pull exact aggregate histories and write the explicitly limited patch."""
    import wandb

    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=180)
    parts = [read_exp232_ccre_trajectory()]
    for arm, (experiment, run_path) in WANDB_RUNS.items():
        run = api.run(run_path)
        history = run.scan_history(keys=["_step", DISTAL_KEY], page_size=1000)
        parts.append(
            history_rows(
                history,
                arm=arm,
                experiment=experiment,
                run_path=run_path,
            )
        )
    trajectory = pd.concat(parts, ignore_index=True)
    for arm, expected in EXPECTED_FINAL.items():
        observed = trajectory[trajectory["arm"] == arm].sort_values(
            ["step", "history_index"]
        )["auprc"].iloc[-1]
        assert abs(observed - expected) < 0.01, (
            f"{arm} final AUPRC {observed:.4f} no longer matches the research record"
        )

    parquet_path = output_dir / "distal_aggregate_trajectories.parquet"
    plot_path = output_dir / "distal_aggregate_trajectories.svg"
    metadata_path = output_dir / "metadata.json"
    trajectory.to_parquet(parquet_path, index=False)
    plot_distal_aggregate(trajectory, plot_path)
    metadata_path.write_text(
        json.dumps(
            {
                "pulled_at": datetime.now(UTC).isoformat(),
                "metric": DISTAL_KEY,
                "aggregate_only": True,
                "soft_metrics_available": False,
                "n_points_by_arm": trajectory.groupby("arm").size().to_dict(),
                "wandb_version": wandb.__version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"trajectory": parquet_path, "plot": plot_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, path in run_distal_patch(args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
