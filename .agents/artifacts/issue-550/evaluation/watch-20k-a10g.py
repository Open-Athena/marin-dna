"""Follow logs, recover completion records, then terminate this task's 20k worker."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MODEL = "dna-exp550-rag46m-five-regions-v1-step-20000"
OUTPUT = Path("/tmp/issue550-20k-completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-key", required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
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
    receipt = None
    with (
        (OUTPUT / "vep.log").open("w", buffering=1) as log,
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
                    assert receipt["model"] == MODEL and receipt["completed"]
                    assert receipt["exit_status"] == 0 and receipt["split"] == "train"
                    assert (
                        receipt["canonical_s3_sizes_verified"]
                        and len(receipt["files"]) == 6
                    )
                    (OUTPUT / "streamed-completion.json").write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
                if line.strip() == "VEP_WORKER_EXIT 0":
                    assert receipt is not None
                    break
            else:
                raise RuntimeError(
                    "Log stream ended without verified completion; retain stopped EBS"
                )
        finally:
            follower.terminate()
            follower.wait(timeout=10)
    # Finish retrieval inside the worker's one-minute shutdown grace.
    names = [
        "step-20000-completed.json",
        "a10g-run-receipt.json",
        "a10g-bf16.yaml",
        "all-tests.log",
        "prepare.log",
        "dry-run.log",
        "batch-sweep/summary.json",
    ]
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
            str(OUTPUT),
        ],
        check=True,
        timeout=40,
    )
    recovered = json.loads((OUTPUT / "step-20000-completed.json").read_text())
    assert recovered == receipt
    command = [
        "aws",
        "ec2",
        "describe-instances",
        "--region",
        "us-east-2",
        "--instance-ids",
        args.instance,
        "--output",
        "json",
    ]
    response = json.loads(subprocess.check_output(command, text=True, timeout=20))
    worker = response["Reservations"][0]["Instances"][0]
    tags = {tag["Key"]: tag["Value"] for tag in worker["Tags"]}
    assert tags["Name"] == "codex-issue550-vep20k-a10g-bf16" and tags["issue"] == "550"
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
    (OUTPUT / "termination-request.json").write_text(result)
    print(
        "20k outputs and logs recovered; termination requested for " + args.instance,
        flush=True,
    )


if __name__ == "__main__":
    main()
