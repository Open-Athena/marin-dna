"""Print or execute commit-pinned SkyPilot commands for exp479."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

REPOSITORY_URL = "https://github.com/Open-Athena/marin-dna.git"
CLUSTER_NAME = "dna-exp479-gh200"
STAGE_CONFIGS = {"preflight": "sky/preflight.yaml"}


def assert_current_clean_commit(commit: str) -> None:
    """Require a clean checkout at the requested 40-character commit."""

    if len(commit) != 40:
        raise ValueError("commit must be a full 40-character SHA")
    root = Path(__file__).resolve().parents[2]
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != commit:
        raise RuntimeError(f"requested commit {commit} differs from checkout {current}")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("commit-pinned launch requires a clean checkout")


def launch_command(stage: str, commit: str, instance_start_unix: int) -> list[str]:
    """Build the self-terminating Lambda GH200 launch command."""

    return [
        "sky",
        "launch",
        "-c",
        CLUSTER_NAME,
        STAGE_CONFIGS[stage],
        "--git-url",
        REPOSITORY_URL,
        "--git-ref",
        commit,
        "--env",
        f"EXPERIMENT_COMMIT={commit}",
        "--env",
        f"EXP479_INSTANCE_START_UNIX={instance_start_unix}",
        "--down",
        "--yes",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGE_CONFIGS))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert_current_clean_commit(args.commit)
    command = launch_command(args.stage, args.commit, int(time.time()))
    print(" ".join(command), flush=True)
    if args.execute:
        subprocess.run(command, check=True, cwd=Path(__file__).parent)


if __name__ == "__main__":
    main()
