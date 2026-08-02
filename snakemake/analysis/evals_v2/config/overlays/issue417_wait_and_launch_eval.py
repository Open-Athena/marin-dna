"""Wait for both terminal exports, run the frozen eval, and write its report.

This is a fail-closed operational handoff for issue #417. It validates the two
complete step-4999 Hugging Face exports, repeats the exact Snakemake dry-run,
launches the bounded auto-downing SkyPilot task, and runs the paired reporter.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from marin_dna_evals.issue417_handoff import (
    parse_sky_status_json,
    validate_hf_export_listing,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
EVAL_DIR = REPO_ROOT / "snakemake/analysis/evals_v2"
RESULTS_PREFIX = "s3://oa-bolinas/snakemake/analysis/evals_v2/"
EXPORTS = {
    "mammals_only": (
        "gs://marin-us-east5/checkpoints/"
        "dna-exp417-cds-mammals-only-p255m-b2m-5k/2026.08.01/hf/step-4999"
    ),
    "combined_vertebrates": (
        "gs://marin-us-east5/checkpoints/"
        "dna-exp417-cds-combined-vertebrates-p255m-b2m-5k/"
        "2026.08.01/hf/step-4999"
    ),
}
MODEL_NAMES = (
    "exp417-cds-mammals-only-step-4999",
    "exp417-cds-combined-vertebrates-step-4999",
)
DATASETS = ("mendelian_traits", "sge")
EVAL_TARGETS = tuple(
    f"results/metrics/{model}/{dataset}.parquet"
    for model in MODEL_NAMES
    for dataset in DATASETS
)
METRIC_PATHS = tuple(f"{RESULTS_PREFIX}{target}" for target in EVAL_TARGETS)
SUMMARY_PATHS = (
    f"{RESULTS_PREFIX}results/comparisons/issue417/summary.json",
    f"{RESULTS_PREFIX}results/comparisons/issue417/summary.md",
)
HANDOFF_LOCK = Path("/tmp/marin-dna-issue417-eval-handoff.lock")
SHARED_HEAVY_LOCK = "/tmp/marin-dna-local-heavy.lock"
SKY_CLUSTER = "dna417-cds-vep"
THREAD_CAPS = (
    "POLARS_MAX_THREADS=2",
    "RAYON_NUM_THREADS=2",
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
)


def _log(message: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{timestamp} {message}", flush=True)


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    _log(f"run cwd={cwd}: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _guarded(command: list[str]) -> list[str]:
    return [
        "flock",
        "-w",
        "3600",
        SHARED_HEAVY_LOCK,
        "env",
        *THREAD_CAPS,
        "nice",
        "-n",
        "10",
        "ionice",
        "-c2",
        "-n7",
        *command,
    ]


def _assert_repo_state(expected_commit: str) -> None:
    head = _run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
    ).stdout.strip()
    assert head == expected_commit, (
        f"HEAD changed: expected {expected_commit}, got {head}"
    )
    status = _run(
        ["git", "status", "--porcelain"],
        capture_output=True,
    ).stdout
    assert not status, f"worktree is dirty:\n{status}"


def _export_sizes(prefix: str) -> dict[str, int] | None:
    result = _run(
        ["gsutil", "ls", "-l", f"{prefix}/**"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return validate_hf_export_listing(prefix, result.stdout)


def _wait_for_exports(*, poll_seconds: int, once: bool) -> bool:
    while True:
        ready: dict[str, dict[str, int]] = {}
        for arm, prefix in EXPORTS.items():
            try:
                sizes = _export_sizes(prefix)
            except AssertionError as error:
                _log(f"{arm} terminal export invalid: {error}")
                sizes = None
            if sizes is not None:
                ready[arm] = sizes
                _log(f"{arm} terminal export verified: {sizes}")
            else:
                _log(f"{arm} terminal export not ready")
        if len(ready) == len(EXPORTS):
            return True
        if once:
            return False
        time.sleep(poll_seconds)


def _all_objects_exist(paths: tuple[str, ...]) -> bool:
    return all(
        _run(
            ["gsutil", "stat", path],
            check=False,
            capture_output=True,
        ).returncode
        == 0
        for path in paths
    )


def _dry_run() -> None:
    _run(
        _guarded(
            [
                "uv",
                "run",
                "snakemake",
                "-n",
                "--",
                *EVAL_TARGETS,
            ]
        ),
        cwd=EVAL_DIR,
    )


def _launch_eval() -> None:
    status = _run(
        ["sky", "status", "-r", "-o", "json", SKY_CLUSTER],
        check=False,
        capture_output=True,
    )
    clusters = parse_sky_status_json(status.stdout)
    assert not clusters, (
        f"Sky cluster already exists; inspect before relaunch: {clusters}"
    )
    snakemake_args = f"-- {' '.join(EVAL_TARGETS)}"
    _run(
        [
            "sky",
            "launch",
            "-y",
            "-c",
            SKY_CLUSTER,
            "--down",
            "snakemake/analysis/evals_v2/sky/run.yaml",
            "--env",
            f"SNAKEMAKE_ARGS={snakemake_args}",
        ]
    )


def _summarize(expected_commit: str) -> None:
    assert _all_objects_exist(METRIC_PATHS), (
        "Sky task returned without all four expected metric parquets"
    )
    _run(
        _guarded(
            [
                "uv",
                "run",
                "--group",
                "genome-s3",
                "python",
                "snakemake/analysis/evals_v2/config/overlays/issue417_summarize_vep.py",
                "--experiment-commit",
                expected_commit,
            ]
        )
    )
    assert _all_objects_exist(SUMMARY_PATHS), "paired summary artifacts are incomplete"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument(
        "--once",
        action="store_true",
        help="check exports once and exit nonzero if they are not complete",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="after validation, run the dry-run, Sky task, and paired reporter",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    assert len(args.expected_commit) == 40 and all(
        character in "0123456789abcdef" for character in args.expected_commit
    ), "expected commit must be a full lowercase Git SHA"
    assert args.poll_seconds > 0, "poll interval must be positive"
    HANDOFF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOCK.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log("another issue #417 evaluation handoff is already running")
            return 2
        lock_handle.write(f"{os.getpid()}\n")
        lock_handle.flush()

        _assert_repo_state(args.expected_commit)
        if not _wait_for_exports(
            poll_seconds=args.poll_seconds,
            once=args.once,
        ):
            return 1
        if not args.launch:
            _log("both terminal exports are complete; launch not requested")
            return 0

        _assert_repo_state(args.expected_commit)
        if not _all_objects_exist(METRIC_PATHS):
            _dry_run()
            _launch_eval()
        else:
            _log("all four metric parquets already exist; skipping Sky launch")
        _summarize(args.expected_commit)
        _log("issue #417 evaluation handoff completed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
