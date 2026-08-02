#!/usr/bin/env python3
"""Plot every issue #417 comparison-model validation loss retained by W&B."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENTITY = "gonzalobenegas"
PROJECT = "marin"
OUTPUT_DIR = Path("plots/output/issue417_validation_loss")
RUNS = {
    "Earlier HAL CDS (exp232)": "dna-exp232-zoonomia-v1-0p25b-v4_cds-v0.1-4224db",
    "Mammals only (exp417)": "dna-exp417-cds-mammals-only-p255m-b2m-5k",
    "Combined vertebrates (exp417)": "dna-exp417-cds-combined-vertebrates-p255m-b2m-5k",
}
MODEL_ORDER = tuple(RUNS)
MODEL_PALETTE = dict(zip(MODEL_ORDER, sns.color_palette("colorblind", 3), strict=True))
MODEL_MARKERS = dict(zip(MODEL_ORDER, ("o", "s", "^"), strict=True))


def _validation_frame() -> tuple[pd.DataFrame, dict[str, int]]:
    api = wandb.Api()
    records: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for model, run_id in RUNS.items():
        by_step: dict[int, float] = {}
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        for row in run.scan_history(keys=["global_step", "eval/loss"], page_size=1_000):
            step = row.get("global_step")
            loss = row.get("eval/loss")
            if step is None or loss is None:
                continue
            by_step[int(step)] = float(loss)
        counts[model] = len(by_step)
        records.extend(
            {"model": model, "step": step, "validation_loss": loss}
            for step, loss in sorted(by_step.items())
        )
    assert counts == {
        "Earlier HAL CDS (exp232)": 8,
        "Mammals only (exp417)": 1,
        "Combined vertebrates (exp417)": 0,
    }, f"W&B coverage changed: {counts}"
    return pd.DataFrame.from_records(records), counts


def main() -> None:
    frame, counts = _validation_frame()
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
        height=5.2,
        aspect=1.55,
        legend=False,
    )
    axis = grid.ax
    axis.set_xlim(0, 5_100)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Validation loss")
    axis.set_title(
        "W&B-retained validation loss\n"
        "Exp417 resumptions disabled W&B after the initial attempts",
        pad=14,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_PALETTE[model],
            marker=MODEL_MARKERS[model],
            linewidth=2.2,
            label=f"{model} ({counts[model]} points)",
        )
        for model in MODEL_ORDER
    ]
    axis.legend(handles=handles, title="W&B coverage", frameon=False, loc="lower left")
    axis.text(
        0.99,
        0.98,
        "Combined exp417: no validation points in W&B\n"
        "Mammals exp417: only step 500 retained",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize="small",
        color="0.3",
    )
    sns.despine(ax=axis)
    grid.figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(OUTPUT_DIR / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(OUTPUT_DIR / "figure.png", bbox_inches="tight", dpi=180)
    plt.close(grid.figure)


if __name__ == "__main__":
    main()
