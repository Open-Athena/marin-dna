"""Recover logs before shutdown and terminate only after all final outputs verify."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MODEL = "dna-exp550-rag46m-five-regions-v1-step-100000"
COHORTS = ("mendelian_traits", "complex_traits", "sge")


def validate_receipt(receipt: dict, model: str = MODEL) -> None:
    expected = {
        f"results/{kind}/{model}/{cohort}.{suffix}"
        for kind, suffix in (
            ("scores", "parquet"),
            ("metrics", "parquet"),
            ("probe", "parquet"),
            ("probe", "joblib"),
            ("probe_metrics", "parquet"),
        )
        for cohort in COHORTS
    }
    assert receipt["model"] == model and receipt["completed"] and receipt["with_probes"]
    assert receipt["exit_status"] == 0 and receipt["split"] == "train"
    assert receipt["canonical_s3_content_sha256_verified"]
    assert set(receipt["files"]) == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[
            f"dna-exp550-rag46m-five-regions-v1-step-{step}" for step in (50000, 100000)
        ],
        default=MODEL,
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    step = int(args.model.rsplit("-", 1)[1])
    args.output.mkdir(parents=True, exist_ok=True)
    # Copy ancillary logs while new logins are permitted, before the run ends.
    names = (
        "a10g-bf16.yaml",
        "all-tests.log",
        "prepare.log",
        "dry-run.log",
        "batch-sweep/summary.json",
    )
    subprocess.run(
        [
            "scp",
            "-i",
            args.ssh_key,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            *[f"{args.host}:/opt/issue550/{name}" for name in names],
            str(args.output),
        ],
        check=True,
        timeout=60,
    )
    receipt = None
    command = [
        "ssh",
        "-T",
        "-i",
        args.ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        args.host,
        "tail -n +1 -F /opt/issue550/vep.log",
    ]
    with (
        (args.output / "vep.log").open("w", buffering=1) as log,
        subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as follower,
    ):
        assert follower.stdout is not None
        try:
            for line in follower.stdout:
                log.write(line)
                if "DEVELOPMENT_VEP_OUTPUTS " in line:
                    receipt = json.loads(line.split("DEVELOPMENT_VEP_OUTPUTS ", 1)[1])
                    validate_receipt(receipt, args.model)
                    (args.output / f"step-{step}-completed.json").write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
                if line.strip() == "VEP_WORKER_EXIT 0":
                    assert receipt is not None
                    break
            else:
                raise RuntimeError(
                    "No verified completion; keep stopped EBS for recovery"
                )
        finally:
            follower.terminate()
            follower.wait(timeout=10)
    # No fresh SSH/SCP connection is opened after completion or shutdown.
    response = json.loads(
        subprocess.check_output(
            [
                "aws",
                "ec2",
                "describe-instances",
                "--region",
                "us-east-2",
                "--instance-ids",
                args.instance,
                "--output",
                "json",
            ],
            text=True,
            timeout=20,
        )
    )
    worker = response["Reservations"][0]["Instances"][0]
    tags = {tag["Key"]: tag["Value"] for tag in worker["Tags"]}
    assert worker["InstanceId"] == args.instance
    assert (
        tags["Name"] == f"codex-issue550-vep{step // 1000}k-a10g-bf16"
        and tags["issue"] == "550"
    )
    result = subprocess.check_output(
        [
            "aws",
            "ec2",
            "terminate-instances",
            "--region",
            "us-east-2",
            "--instance-ids",
            args.instance,
            "--output",
            "json",
        ],
        text=True,
        timeout=20,
    )
    (args.output / "termination-request.json").write_text(result)
    print("Final outputs and logs preserved; worker termination requested.", flush=True)


if __name__ == "__main__":
    main()
