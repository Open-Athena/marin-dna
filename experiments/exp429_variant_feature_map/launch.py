"""Print or execute the pinned Sky commands for experiment 429."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from sample_panel import assert_current_commit

CPU_CLUSTER = "exp429-cpu"
GPU_CLUSTER = "exp429-gpu"
STAGE_CONFIG = {
    "panel": "sky.cpu.yaml",
    "extract": "sky.gpu.yaml",
    "analyze": "sky.analysis.yaml",
}


def sky_command(stage: str, commit: str) -> list[str]:
    assert stage in STAGE_CONFIG
    assert len(commit) == 40
    cluster = GPU_CLUSTER if stage == "extract" else CPU_CLUSTER
    # Always use launch, including for a warm cluster: `sky exec` skips the
    # YAML setup phase, which is responsible for checking out the pinned
    # experiment commit and installing its environment.
    command = ["sky", "launch", "-c", cluster]
    command.extend([STAGE_CONFIG[stage], "--env", f"EXPERIMENT_COMMIT={commit}"])
    command.extend(["--yes"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGE_CONFIG))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert_current_commit(args.commit)
    command = sky_command(args.stage, args.commit)
    print(" ".join(command), flush=True)
    if args.execute:
        subprocess.run(command, check=True, cwd=Path(__file__).parent)


if __name__ == "__main__":
    main()
