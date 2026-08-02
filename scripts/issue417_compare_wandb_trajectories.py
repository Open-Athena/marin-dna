#!/usr/bin/env python3
"""Compare the full available W&B trajectories for issue #417 and exp232 CDS.

The exp417 resumes intentionally disabled W&B after checkpoint-teardown races,
so "full" here means every point retained by W&B. The output makes each run's
coverage explicit rather than extrapolating the missing exp417 tail.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import wandb

ENTITY = "gonzalobenegas"
PROJECT = "marin"

RUNS = {
    "exp232_zoonomia_v4_cds": "dna-exp232-zoonomia-v1-0p25b-v4_cds-v0.1-4224db",
    "exp417_mammals_only": "dna-exp417-cds-mammals-only-p255m-b2m-5k",
    "exp417_combined_vertebrates": "dna-exp417-cds-combined-vertebrates-p255m-b2m-5k",
}

METRICS = (
    "train/loss",
    "eval/loss",
    "eval/macro_loss",
    "optim/learning_rate",
    "grad/norm/total",
    "throughput/examples_per_second",
)

CHECKPOINT_STEPS = (100, 500, 1_000, 2_000, 3_000, 4_000, 4_999)

CONFIG_KEYWORDS = re.compile(
    r"model|optimizer|trainer|data_seed|loss|batch|seq|seed|mp|gradient|"
    r"parallel|shuffle|tokenizer|format|checkpoint|hf_save",
    re.IGNORECASE,
)


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}

    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_config(child, path))
    return flattened


def _fetch_series(run: Any, metric: str) -> list[tuple[int, float]]:
    """Fetch every retained point where both global_step and metric exist."""
    by_step: dict[int, float] = {}
    for row in run.scan_history(keys=["global_step", metric], page_size=1_000):
        step = row.get("global_step")
        value = row.get(metric)
        if step is None or value is None:
            continue
        by_step[int(step)] = float(value)
    return sorted(by_step.items())


def _nearest(points: Sequence[tuple[int, float]], step: int) -> tuple[int, float] | None:
    if not points:
        return None
    return min(points, key=lambda point: abs(point[0] - step))


def _series_summary(points: Sequence[tuple[int, float]]) -> dict[str, Any]:
    if not points:
        return {
            "count": 0,
            "first_step": None,
            "last_step": None,
            "first_value": None,
            "last_value": None,
            "minimum": None,
            "last_100_mean": None,
            "at_checkpoints": {},
        }

    tail = points[-100:]
    return {
        "count": len(points),
        "first_step": points[0][0],
        "last_step": points[-1][0],
        "first_value": points[0][1],
        "last_value": points[-1][1],
        "minimum": min(value for _, value in points),
        "last_100_mean": sum(value for _, value in tail) / len(tail),
        "at_checkpoints": {
            str(step): {"observed_step": point[0], "value": point[1]}
            for step in CHECKPOINT_STEPS
            if (point := _nearest(points, step)) is not None
        },
    }


def _format_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Issue #417 W&B trajectory comparison",
        "",
        (
            "Every W&B-retained point is included. The exp417 resumes disabled W&B, "
            "so their reported coverage ends before training step 4999."
        ),
        "",
        "## Train loss",
        "",
        "| Run | W&B coverage | step 100 | step 500 | step 1000 | step 2000 | step 3000 | step 4000 | step 4999 | last-100 mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for label, run_summary in summary["runs"].items():
        train = run_summary["metrics"]["train/loss"]
        checkpoint_values = []
        for step in CHECKPOINT_STEPS:
            point = train["at_checkpoints"].get(str(step))
            if point is None or abs(point["observed_step"] - step) > 1:
                checkpoint_values.append("—")
            else:
                checkpoint_values.append(_format_value(point["value"]))
        coverage = f"{train['first_step']}–{train['last_step']} ({train['count']:,})"
        lines.append(
            "| "
            + " | ".join(
                [label, coverage, *checkpoint_values, _format_value(train["last_100_mean"])]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Evaluation-loss coverage",
            "",
            "| Run | eval/loss points | first step | last step | last value |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, run_summary in summary["runs"].items():
        evaluation = run_summary["metrics"]["eval/loss"]
        lines.append(
            f"| {label} | {evaluation['count']:,} | {evaluation['first_step'] or '—'} | "
            f"{evaluation['last_step'] or '—'} | {_format_value(evaluation['last_value'])} |"
        )

    path.write_text("\n".join(lines) + "\n")


def compare(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    summary: dict[str, Any] = {
        "entity": ENTITY,
        "project": PROJECT,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "runs": {},
    }
    configs: dict[str, Any] = {}

    csv_path = output_dir / "trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "run_id", "metric", "step", "value"))
        writer.writeheader()

        for label, run_id in RUNS.items():
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            metric_summaries: dict[str, Any] = {}
            for metric in METRICS:
                points = _fetch_series(run, metric)
                metric_summaries[metric] = _series_summary(points)
                writer.writerows(
                    {
                        "run": label,
                        "run_id": run_id,
                        "metric": metric,
                        "step": step,
                        "value": value,
                    }
                    for step, value in points
                )

            summary["runs"][label] = {
                "run_id": run_id,
                "name": run.name,
                "url": run.url,
                "state": run.state,
                "metrics": metric_summaries,
            }

            flat_config = _flatten_config(run.config)
            configs[label] = {
                key: value for key, value in sorted(flat_config.items()) if CONFIG_KEYWORDS.search(key)
            }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "configs.json").write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, output_dir / "summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/issue417/wandb_trajectories"),
        help="Directory for trajectory.csv, summary.json, summary.md, and configs.json.",
    )
    args = parser.parse_args()
    summary = compare(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
