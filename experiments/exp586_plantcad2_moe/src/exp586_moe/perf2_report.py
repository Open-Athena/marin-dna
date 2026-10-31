"""Download complete bounded profile histories for repeatable comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import wandb


def _plain(value):
    if hasattr(value, "keys"):
        # W&B SummarySubDict supports keys() but does not support iteration.
        return {key: _plain(value[key]) for key in value.keys()}  # noqa: SIM118
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for run in wandb.Api().runs(
        "eric-czech/marin", filters={"group": "exp586-plantcad2-moe-perf2"}
    ):
        records[run.id] = {
            "state": run.state,
            "summary": _plain(run.summary),
            "history": list(run.scan_history()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    baseline_name = (
        "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-combined-a1"
    )
    baseline = {
        row["train/step"]: row["train/loss"]
        for row in records[baseline_name]["history"]
        if row.get("train/loss") is not None
    }
    for name, record in sorted(records.items()):
        summary = record["summary"]
        result = {
            "run": name.removeprefix(
                "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-"
            ),
            "state": record["state"],
            "phase": summary.get("profiling/phase"),
            "median_step": summary.get("profiling/median_step_seconds"),
            "median_iteration": summary.get("profiling/median_iteration_seconds"),
        }
        history = [
            row for row in record["history"] if row.get("train/loss") is not None
        ]
        differences = np.asarray(
            [
                row["train/loss"] - baseline[row["train/step"]]
                for row in history
                if row["train/step"] in baseline
            ]
        )
        if len(differences):
            result.update(
                samples=len(differences),
                loss_rmse=float(np.sqrt(np.mean(differences**2))),
                loss_mae=float(np.mean(np.abs(differences))),
                loss_signed_mean=float(np.mean(differences)),
                loss_max_abs=float(np.max(np.abs(differences))),
            )
        rows = [row for row in history if row.get("train/step", 0) >= 30]
        if len(rows) > 1:
            result["median_timestamp_interval"] = float(
                np.median(np.diff([row["_timestamp"] for row in rows]))
            )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
