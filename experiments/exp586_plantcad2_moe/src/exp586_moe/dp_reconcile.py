"""Record verified terminal facts for one explicitly selected profile."""

import argparse
import datetime as dt
import json
import os
import subprocess

import fsspec
import wandb

from exp586_moe.dp_control import DATABASE, _mutate, configure_storage
from exp586_moe.dp_profile import VARIANTS, DPConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--attempt", default="a1")
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nodes", type=int, choices=[1, 8], default=1)
    parser.add_argument("--ep-reference", choices=["best", "live"], default="best")
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--end-to-end-updates", type=int, default=60)
    parser.add_argument(
        "--end-to-end-order",
        choices=["dp-first", "ep-first", "balanced"],
        default="dp-first",
    )
    config = DPConfig(**vars(parser.parse_args()))
    run = wandb.Api(api_key=os.environ["WANDB_API_KEY"]).run(
        f"eric-czech/marin/{config.run_id}"
    )
    job = config.job_name
    result = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "--extra",
            "cpu",
            "iris",
            "--cluster",
            "marin",
            "job",
            "wait",
            f"/eczech/{job}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    terminal = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if terminal not in ("succeeded", "failed", "killed", "cancelled"):
        raise ValueError(f"Root job is not terminal: {terminal}")
    run = wandb.Api(api_key=os.environ["WANDB_API_KEY"]).run(
        f"eric-czech/marin/{config.run_id}"
    )
    configure_storage()
    now = dt.datetime.now(dt.UTC).isoformat()
    _mutate(
        "observe",
        str(DATABASE),
        "--dispatch-id",
        job,
        "--observed-at",
        now,
        "--wandb-state",
        run.state,
        "--run-progress",
        str(float(run.summary.get("run_progress", 0))),
        "--iris-running",
        "false",
    )
    _mutate(
        "dispatch-end",
        str(DATABASE),
        "--dispatch-id",
        job,
        "--outcome",
        terminal,
    )
    if terminal != "succeeded" or run.summary.get("profiling/phase") != "complete":
        print(
            json.dumps(
                {
                    "terminal": terminal,
                    "wandb_state": run.state,
                    "phase": run.summary.get("profiling/phase"),
                    "error": run.summary.get("profiling/error"),
                }
            )
        )
        return
    fs = fsspec.filesystem("s3")
    result = json.loads(
        fs.cat((config.artifact_root + "/result.json").removeprefix("s3://"))
    )
    if result["profiling/updates"] != config.updates:
        raise ValueError("Profile did not complete its expected updates")
    steps = [
        row["train/step"] for row in run.scan_history(keys=["train/step", "train/loss"])
    ]
    if steps != list(range(1, config.updates + 1)):
        raise ValueError("Profile history is incomplete or contains restarted updates")
    _mutate(
        "trial-complete",
        str(DATABASE),
        "--trial-id",
        config.run_id,
        "--checkpoint-verified-at",
        now,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
