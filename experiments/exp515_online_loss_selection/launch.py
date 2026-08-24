"""Print or execute the commit-pinned issue #515 SkyPilot launch."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path

REPOSITORY_URL = "https://github.com/Open-Athena/marin-dna.git"
CLUSTER_NAME = "dna-exp515-online-selection-a100"
CASE_AUDIT = ".agents/artifacts/515-online-loss-selection/case-distribution-audit.json"


def _clean_commit(commit: str) -> None:
    root = Path(__file__).resolve().parents[2]
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if len(commit) != 40 or commit != current or status:
        raise RuntimeError("paid launch requires the exact clean 40-character commit")
    if not (root / CASE_AUDIT).exists():
        raise FileNotFoundError(
            f"paid launch requires committed case audit {CASE_AUDIT}"
        )


def _execution_environment() -> dict[str, str]:
    environment = dict(os.environ)
    adc_path = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    if not adc_path.exists():
        raise FileNotFoundError("Google application-default credentials are missing")
    environment["GOOGLE_ADC_JSON_BASE64"] = base64.b64encode(
        adc_path.read_bytes()
    ).decode()
    if not environment.get("AWS_ACCESS_KEY_ID") or not environment.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        exported = subprocess.run(
            ["aws", "configure", "export-credentials", "--format", "process"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(exported.stdout)
        environment["AWS_ACCESS_KEY_ID"] = payload["AccessKeyId"]
        environment["AWS_SECRET_ACCESS_KEY"] = payload["SecretAccessKey"]
        if payload.get("SessionToken"):
            environment["AWS_SESSION_TOKEN"] = payload["SessionToken"]
    return environment


def launch_command(
    commit: str,
    run_id: str,
    *,
    retry_until_up: bool,
    include_aws_session_token: bool = False,
) -> list[str]:
    command = [
        "sky",
        "launch",
        "-c",
        CLUSTER_NAME,
        "sky/pilot.yaml",
        "--git-url",
        REPOSITORY_URL,
        "--git-ref",
        commit,
        "--env",
        f"EXPERIMENT_COMMIT={commit}",
        "--env",
        f"EXP515_RUN_ID={run_id}",
        "--env",
        f"EXP515_INSTANCE_START_UNIX={int(time.time())}",
        "--secret",
        "GOOGLE_ADC_JSON_BASE64",
        "--secret",
        "AWS_ACCESS_KEY_ID",
        "--secret",
        "AWS_SECRET_ACCESS_KEY",
    ]
    if include_aws_session_token:
        command.extend(["--secret", "AWS_SESSION_TOKEN"])
    if retry_until_up:
        command.append("--retry-until-up")
    command.extend(["--down", "--yes"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-until-up", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id or f"{args.commit[:12]}-{int(time.time())}"
    _clean_commit(args.commit)
    environment = _execution_environment() if args.execute else dict(os.environ)
    command = launch_command(
        args.commit,
        run_id,
        retry_until_up=args.retry_until_up,
        include_aws_session_token=bool(environment.get("AWS_SESSION_TOKEN")),
    )
    print(" ".join(command))
    if args.execute:
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent,
            check=True,
            env=environment,
        )


if __name__ == "__main__":
    main()
