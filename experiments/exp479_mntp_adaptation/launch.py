"""Print or execute commit-pinned SkyPilot commands for exp479."""

from __future__ import annotations

import argparse
import netrc
import os
import subprocess
import time
from pathlib import Path

REPOSITORY_URL = "https://github.com/Open-Athena/marin-dna.git"
CLUSTER_NAME = "dna-exp479-gh200"
STAGE_CONFIGS = {
    "preflight": "sky/preflight.yaml",
    "pilot": "sky/pilot.yaml",
    "diagnostics": "sky/diagnostics.yaml",
    "audit": "sky/audit.yaml",
    "stability": "sky/stability.yaml",
    "dependency": "sky/dependency.yaml",
    "calibration": "sky/calibration.yaml",
    "longrun": "sky/longrun.yaml",
    "loss-normalization": "sky/loss-normalization.yaml",
    "source-validation": "sky/source-validation.yaml",
}
HF_REPO_ID = "marin-dna/marin-dna-exp479-mntp-m5.1"


def execution_environment(stage: str) -> dict[str, str]:
    """Load existing local credentials into memory for Sky secret forwarding."""

    environment = dict(os.environ)
    hf_stages = {"pilot", "diagnostics", "audit", "stability", "dependency", "calibration"}
    wandb_stages = {
        "pilot",
        "audit",
        "stability",
        "dependency",
        "calibration",
        "longrun",
        "loss-normalization",
        "source-validation",
    }
    if stage not in hf_stages | wandb_stages:
        return environment
    if stage in hf_stages and not environment.get("HF_TOKEN"):
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            environment["HF_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    if stage in wandb_stages and not environment.get("WANDB_API_KEY"):
        authentication = netrc.netrc().authenticators("api.wandb.ai")
        if authentication is not None:
            environment["WANDB_API_KEY"] = authentication[2]
    if stage in {"longrun", "loss-normalization", "source-validation"}:
        required = ("WANDB_API_KEY",)
    elif stage in wandb_stages:
        required = ("HF_TOKEN", "WANDB_API_KEY")
    else:
        required = ("HF_TOKEN",)
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise RuntimeError(f"paid stage lacks required local credentials: {missing}")
    return environment


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


def launch_command(
    stage: str,
    commit: str,
    instance_start_unix: int,
    hf_repo_id: str = HF_REPO_ID,
    dry_run: bool = False,
    resume_hf_repo_id: str | None = None,
    checkpoint_upload_steps: tuple[int, ...] = (),
    prior_cost_usd: float = 0.0,
    retry_until_up: bool = False,
) -> list[str]:
    """Build the self-terminating Lambda GH200 launch command."""

    command = [
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
    ]
    if stage in {"pilot", "diagnostics", "audit", "stability", "dependency", "calibration"}:
        command.extend(
            [
                "--env",
                f"HF_REPO_ID={hf_repo_id}",
                "--secret",
                "HF_TOKEN",
            ]
        )
        if stage in {"pilot", "audit", "stability", "dependency", "calibration"}:
            command.extend(["--secret", "WANDB_API_KEY"])
        if resume_hf_repo_id is not None:
            command.extend(["--env", f"RESUME_HF_REPO_ID={resume_hf_repo_id}"])
        if checkpoint_upload_steps:
            steps = " ".join(map(str, checkpoint_upload_steps))
            command.extend(["--env", f"CHECKPOINT_UPLOAD_STEPS={steps}"])
    if stage in {"longrun", "loss-normalization", "source-validation"}:
        command.extend(["--secret", "WANDB_API_KEY"])
    if prior_cost_usd:
        command.extend(["--env", f"EXP479_PRIOR_COST_USD={prior_cost_usd}"])
    if retry_until_up:
        command.append("--retry-until-up")
    command.extend(["--down", "--yes"])
    if dry_run:
        command.append("--dryrun")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGE_CONFIGS))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--resume-hf-repo-id")
    parser.add_argument("--checkpoint-upload-steps", type=int, nargs="*", default=())
    parser.add_argument("--prior-cost-usd", type=float, default=0.0)
    parser.add_argument("--retry-until-up", action="store_true")
    parser.add_argument("--model-card-reviewed", action="store_true")
    args = parser.parse_args()
    assert_current_clean_commit(args.commit)
    if args.stage == "pilot" and args.execute and not args.dry_run and not args.model_card_reviewed:
        raise RuntimeError("paid pilot requires explicit human model-card review")
    command = launch_command(
        args.stage,
        args.commit,
        int(time.time()),
        hf_repo_id=args.hf_repo_id,
        dry_run=args.dry_run,
        resume_hf_repo_id=args.resume_hf_repo_id,
        checkpoint_upload_steps=tuple(args.checkpoint_upload_steps),
        prior_cost_usd=args.prior_cost_usd,
        retry_until_up=args.retry_until_up,
    )
    print(" ".join(command), flush=True)
    if args.execute:
        subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).parent,
            env=execution_environment(args.stage),
        )


if __name__ == "__main__":
    main()
