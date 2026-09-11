# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3==1.41.5"]
# ///
"""Watch one registered VEP job and publish its six completed result objects.

The local process reads small status/receipt messages every fifteen minutes.
A bounded Iris CPU job copies verified bytes from CoreWeave to the canonical AWS
objects using object-scoped PUT URLs; no reusable AWS credentials leave the VM.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from urllib.parse import unquote, urlsplit
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

MODEL = "dna-exp550-rag46m-five-regions-v1-step-10000"
COHORTS = {"mendelian_traits": 16140, "complex_traits": 11630, "sge": 23853}
EXPECTED = {
    f"results/{kind}/{MODEL}/{name}.parquet"
    for kind in ("scores", "metrics")
    for name in COHORTS
}
AWS_PREFIX = "snakemake/analysis/evals_v2/"
IRIS = "/tmp/issue550-iris-client/bin/iris"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def source_prefix(commit: str) -> str:
    if not HEX40.fullmatch(commit):
        raise ValueError("Invalid consumer commit")
    return f"marin/MarinDNA/exp550_rag_five_regions/evaluation-staging/{MODEL}/{commit}"


def validate_receipt(receipt: dict, commit: str) -> None:
    assert receipt["model"] == MODEL and receipt["consumer_commit"] == commit
    assert receipt["split"] == "train" and receipt["cohort_sizes"] == COHORTS
    assert (
        receipt["checkpoint_sha256"]
        == "a4fd7d562c61aade4f86d7a3349c5894d3a18c93357d797b6a1e3cd8171d3dad"
    )
    assert (
        receipt["harness_sha256"]
        == "6631d35f9ae0afc754c623a2e3960682e8a834a28001906279745882f99cb73b"
    )
    assert receipt["completed"] and receipt["scores_complete"]
    assert (
        receipt["staging_prefix"] == f"s3://marin-us-east-02a/{source_prefix(commit)}/"
    )
    assert receipt["canonical_prefix"] == "s3://oa-bolinas/" + AWS_PREFIX
    assert set(receipt["files"]) == EXPECTED
    for item in receipt["files"].values():
        assert isinstance(item["bytes"], int) and 0 < item["bytes"] <= 256 * 1024**2
        assert HEX64.fullmatch(item["sha256"])


def checksum(sha256: str) -> str:
    return base64.b64encode(bytes.fromhex(sha256)).decode()


def existing_matches(client, path: str, item: dict) -> bool:
    try:
        head = client.head_object(
            Bucket="oa-bolinas", Key=AWS_PREFIX + path, ChecksumMode="ENABLED"
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    if head["ContentLength"] != item["bytes"] or head.get("ChecksumSHA256") != checksum(
        item["sha256"]
    ):
        raise ValueError(
            f"Existing canonical output differs or lacks a verified checksum: {path}"
        )
    return True


def worker(commit: str) -> None:
    client = boto3.client("s3", config=Config(s3={"addressing_style": "virtual"}))
    prefix = source_prefix(commit)
    receipt = json.loads(
        client.get_object(Bucket="marin-us-east-02a", Key=f"{prefix}/receipt.json")[
            "Body"
        ].read()
    )
    validate_receipt(receipt, commit)
    targets = json.loads(os.environ.pop("ISSUE550_PUBLISH_TARGETS"))
    assert 0 < len(targets) <= len(EXPECTED)
    assert len({item["path"] for item in targets}) == len(targets)
    for target in targets:
        path = target["path"]
        assert path in EXPECTED
        url = urlsplit(target["url"])
        assert url.scheme == "https"
        assert url.netloc in {
            "oa-bolinas.s3.us-east-2.amazonaws.com",
            "oa-bolinas.s3.amazonaws.com",
        }
        assert unquote(url.path) == "/" + AWS_PREFIX + path
        meta = receipt["files"][path]
        assert target["sha256"] == meta["sha256"] and target["bytes"] == meta["bytes"]
        headers = {
            "Content-Length": str(meta["bytes"]),
            "If-None-Match": "*",
            "x-amz-meta-sha256": meta["sha256"],
            "x-amz-checksum-sha256": checksum(meta["sha256"]),
        }
        with tempfile.TemporaryDirectory(prefix="issue550-publish-") as folder:
            local = Path(folder) / "result.parquet"
            client.download_file(
                "marin-us-east-02a",
                f"{prefix}/{path}",
                str(local),
                Config=TransferConfig(use_threads=False),
            )
            assert local.stat().st_size == meta["bytes"]
            with local.open("rb") as handle:
                assert (
                    hashlib.file_digest(handle, "sha256").hexdigest() == meta["sha256"]
                )
            try:
                with local.open("rb") as data:
                    request = urllib.request.Request(
                        target["url"], data=data, headers=headers, method="PUT"
                    )
                    with urllib.request.urlopen(request, timeout=300) as response:
                        assert response.status == 200
            except Exception as exc:
                raise RuntimeError(
                    f"Canonical upload failed for {path}: {type(exc).__name__}"
                ) from None
        print("CANONICAL_OBJECT_WRITTEN " + path, flush=True)


def iris_command(*args: str) -> str:
    result = subprocess.run(
        [IRIS, "--cluster", "marin", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return result.stdout


def task_state(job: str) -> str:
    return json.loads(
        iris_command("rpc", "controller", "get-task-status", "--task-id", job + "/0")
    )["task"]["state"]


def completed_receipt(job: str, commit: str) -> dict | None:
    logs = iris_command(
        "job",
        "logs",
        job,
        "--substring",
        "VEP_DEVELOPMENT_COMPLETE",
        "--max-lines",
        "5",
    )
    matches = [
        line.split("VEP_DEVELOPMENT_COMPLETE ", 1)[1]
        for line in logs.splitlines()
        if "VEP_DEVELOPMENT_COMPLETE " in line
    ]
    if not matches:
        return None
    receipt = json.loads(matches[-1])
    validate_receipt(receipt, commit)
    return receipt


def submit_private(command: list[str], root: Path) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, timeout=120
        )
    except (subprocess.TimeoutExpired, OSError):
        raise RuntimeError(
            "Publication submission failed or timed out; inspect job status before retrying"
        ) from None
    if result.returncode:
        raise RuntimeError(
            "Publication job submission failed; completed outputs remain in CoreWeave"
        )
    return result


def watch(
    job: str, commit: str, receipt_path: Path, *, stage_only: bool = False
) -> None:
    source_prefix(commit)
    if not job.startswith("/gonzalo/dna-exp550-10k-vep-h100-") or not re.fullmatch(
        r"/[a-z0-9/-]+", job
    ):
        raise ValueError("Unexpected job")
    root = Path(__file__).resolve().parents[4]
    publisher_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if not HEX40.fullmatch(publisher_commit):
        raise ValueError("Invalid publisher commit")
    checked_source = subprocess.check_output(
        [
            "git",
            "show",
            f"{publisher_commit}:.agents/artifacts/issue-550/evaluation/publish-shared-results.py",
        ],
        cwd=root,
    )
    if checked_source != Path(__file__).read_bytes():
        raise ValueError("Publisher must run from its committed source")
    deadline = time.monotonic() + 9 * 3600
    while time.monotonic() < deadline:
        receipt = completed_receipt(job, commit)
        if receipt is not None:
            break
        if task_state(job) in {
            "TASK_STATE_FAILED",
            "TASK_STATE_KILLED",
            "TASK_STATE_SUCCEEDED",
        }:
            receipt = completed_receipt(job, commit)
            if receipt is not None:
                break
            raise RuntimeError(
                "Evaluation ended without a completed six-object receipt; inspect retained Iris logs"
            )
        time.sleep(900)
    else:
        raise TimeoutError("Evaluation watch reached its nine-hour deadline")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if stage_only:
        print("VEP_COMPLETE_STAGED " + str(receipt_path), flush=True)
        return
    aws = boto3.client(
        "s3",
        region_name="us-east-2",
        endpoint_url="https://s3.us-east-2.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
    targets = []
    for path, item in sorted(receipt["files"].items()):
        if existing_matches(aws, path, item):
            continue
        url = aws.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": "oa-bolinas",
                "Key": AWS_PREFIX + path,
                "ContentLength": item["bytes"],
                "IfNoneMatch": "*",
                "Metadata": {"sha256": item["sha256"]},
                "ChecksumSHA256": checksum(item["sha256"]),
            },
            ExpiresIn=3600,
            HttpMethod="PUT",
        )
        targets.append({"path": path, "url": url, **item})
    if targets:
        # Only a normal bounded CPU job is created; the GPU job can exit as soon
        # as its own durable outputs complete. The source hash is a shell-safe
        # Git hex ID and all signed URLs are in the runtime environment.
        script = f"""set -eu
curl -fLsS https://astral.sh/uv/0.11.31/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
export PATH=/usr/local/bin:$PATH
git -C /app init
git -C /app fetch --depth 1 https://github.com/Open-Athena/marin-dna.git {publisher_commit}
git -C /app reset --hard FETCH_HEAD
uv run --locked --script /app/.agents/artifacts/issue-550/evaluation/publish-shared-results.py --worker --consumer-commit {commit}
"""
        command = [
            IRIS,
            "--cluster",
            "marin",
            "job",
            "run",
            "--no-wait",
            "--no-sync",
            "--user",
            "gonzalo",
            "--job-name",
            f"dna-exp550-10k-publish-{int(time.time())}",
            "--target-cluster",
            "cw-us-east-02a",
            "--cpu",
            "1",
            "--memory",
            "2G",
            "--disk",
            "5G",
            "--timeout",
            "1200",
            "--max-retries",
            "0",
            "--exclude",
            r"^(?!\.agents/artifacts/issue-550/evaluation/publish-shared-results\.py(?:\.lock)?$)",
            "-e",
            "ISSUE550_PUBLISH_TARGETS",
            json.dumps(targets),
            "--",
            "bash",
            "-lc",
            script,
        ]
        # Never expose secret-bearing command arguments through an exception.
        submitted = submit_private(command, root)
        publish_job = next(
            line
            for line in submitted.stdout.splitlines()
            if line.startswith("/gonzalo/dna-exp550-10k-publish-")
        )
        print("PUBLISH_JOB " + publish_job, flush=True)
        publish_deadline = time.monotonic() + 45 * 60
        while time.monotonic() < publish_deadline:
            if all(
                existing_matches(aws, path, item)
                for path, item in receipt["files"].items()
            ):
                break
            if task_state(publish_job) in {
                "TASK_STATE_FAILED",
                "TASK_STATE_KILLED",
                "TASK_STATE_SUCCEEDED",
            }:
                if all(
                    existing_matches(aws, path, item)
                    for path, item in receipt["files"].items()
                ):
                    break
                raise RuntimeError(
                    "Publication job ended before all canonical objects verified"
                )
            time.sleep(900)
        else:
            raise TimeoutError(
                "Publication exceeded its bounded wait; staged outputs are retained"
            )
    receipt["canonical_published"] = True
    receipt["canonical_verified_at_unix"] = time.time()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("CANONICAL_PUBLICATION_VERIFIED " + str(receipt_path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--job")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.stage_only:
            parser.error("--stage-only applies only to the read-only watcher")
        worker(args.consumer_commit)
    else:
        if args.job is None or args.receipt is None:
            parser.error("watch mode requires --job and --receipt")
        watch(args.job, args.consumer_commit, args.receipt, stage_only=args.stage_only)
