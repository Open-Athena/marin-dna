#!/usr/bin/env python3
"""Plot both issue #417 validation-loss trajectories recovered from Iris logs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path("plots/output/issue417_validation_loss")
EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "experiments/exp417_vertebrate_cds"
)
IRIS_CLI = EXPERIMENT_DIR / ".venv/bin/iris"

MODEL_ORDER = ("Mammals only", "Combined vertebrates")
MODEL_PALETTE = dict(zip(MODEL_ORDER, sns.color_palette("colorblind", 2), strict=True))
MODEL_MARKERS = dict(zip(MODEL_ORDER, ("o", "s"), strict=True))
CHECKPOINT_STEPS = (
    500,
    1_000,
    1_500,
    2_000,
    2_500,
    3_000,
    3_500,
    4_000,
    4_500,
    4_999,
)

IRIS_JOBS = {
    "Mammals only": (
        "/ubuntu/dna-exp417-cds-mammals-only",
        "/ubuntu/dna-exp417-cds-mammals-only-r2",
        "/ubuntu/dna-exp417-cds-mammals-only-r3",
        "/ubuntu/dna-exp417-cds-mammals-only-r4",
    ),
    "Combined vertebrates": (
        "/ubuntu/dna-exp417-cds-combined-vertebrates-r2",
        "/ubuntu/dna-exp417-cds-combined-vertebrates-r3",
        "/ubuntu/dna-exp417-cds-combined-vertebrates-r5",
        "/ubuntu/dna-exp417-cds-combined-vertebrates-r6",
    ),
}

SAVE_STEP_PATTERN = re.compile(
    r"levanter\.checkpoint Saving checkpoint at step ([0-9]+)\."
)
EVAL_LOSS_PATTERN = re.compile(r"levanter\.eval eval loss: ([0-9]+(?:\.[0-9]+)?)")


def _fetch_iris_log(job_id: str) -> str:
    assert IRIS_CLI.is_file(), (
        f"missing {IRIS_CLI}; run `uv sync` in {EXPERIMENT_DIR} before plotting"
    )
    result = subprocess.run(
        [
            str(IRIS_CLI),
            "--cluster=marin",
            "job",
            "logs",
            job_id,
            "--max-lines",
            "20000",
            "--tail",
        ],
        cwd=EXPERIMENT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _parse_validation_losses(log: str, job_id: str) -> dict[int, float]:
    current_step: int | None = None
    losses: dict[int, float] = {}
    for line in log.splitlines():
        if step_match := SAVE_STEP_PATTERN.search(line):
            current_step = int(step_match.group(1))
            continue
        if loss_match := EVAL_LOSS_PATTERN.search(line):
            assert current_step is not None, (
                f"{job_id}: eval loss without checkpoint step"
            )
            loss = float(loss_match.group(1))
            previous = losses.setdefault(current_step, loss)
            assert previous == loss, (
                f"{job_id}: conflicting losses at step {current_step}: "
                f"{previous} vs {loss}"
            )
    return losses


def _validation_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model, job_ids in IRIS_JOBS.items():
        model_losses: dict[int, float] = {}
        for job_id in job_ids:
            losses = _parse_validation_losses(_fetch_iris_log(job_id), job_id)
            for step, loss in losses.items():
                previous = model_losses.setdefault(step, loss)
                assert previous == loss, (
                    f"{model}: conflicting Iris losses at step {step}: "
                    f"{previous} vs {loss}"
                )
        assert tuple(sorted(model_losses)) == CHECKPOINT_STEPS, (
            f"{model}: expected checkpoints {CHECKPOINT_STEPS}, got "
            f"{tuple(sorted(model_losses))}"
        )
        records.extend(
            {"model": model, "step": step, "validation_loss": model_losses[step]}
            for step in CHECKPOINT_STEPS
        )
    frame = pd.DataFrame.from_records(records)
    assert frame.groupby("model").size().to_dict() == {
        "Combined vertebrates": 10,
        "Mammals only": 10,
    }
    return frame


def main() -> None:
    frame = _validation_frame()
    sns.set_theme(style="ticks", context="talk")
    grid = sns.relplot(
        data=frame,
        x="step",
        y="validation_loss",
        hue="model",
        hue_order=MODEL_ORDER,
        style="model",
        style_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        markers=MODEL_MARKERS,
        dashes=False,
        kind="line",
        linewidth=2.2,
        markersize=8,
        height=5.7,
        aspect=1.55,
    )
    axis = grid.ax
    axis.set_xlim(400, 5_100)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Validation loss")
    axis.set_title("")
    grid.legend.set_title("")
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=axis)
    grid.figure.suptitle("Validation loss across training checkpoints", y=0.98)
    grid.figure.text(
        0.5,
        0.015,
        "Each arm's own validation corpus · compare trajectory shape, not absolute level",
        ha="center",
        fontsize=11,
    )
    grid.figure.subplots_adjust(top=0.76, bottom=0.22)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(OUTPUT_DIR / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(OUTPUT_DIR / "figure.png", bbox_inches="tight", dpi=180)
    plt.close(grid.figure)


if __name__ == "__main__":
    main()
