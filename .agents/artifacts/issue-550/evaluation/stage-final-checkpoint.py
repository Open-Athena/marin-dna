# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3==1.41.5", "PyYAML==6.0.3"]
# ///
"""Mirror the registered final export into its canonical S3 checkpoint cache.

This one-off transport avoids importing Torch/Snakemake on the shared VM.
It uses normal gcloud and boto3 credential providers, never transfers credentials,
and requires --apply for writes. Existing objects must match the GCS bytes.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import resource
import signal
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[4]
MODEL = "dna-exp550-rag46m-five-regions-v1-step-100000"
BUCKET = "oa-bolinas"
PREFIX = f"snakemake/analysis/evals_v2/results/checkpoints/{MODEL}/"
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
GIB = 1024**3
ESTIMATED_WORKING_SET = 400 * 1024**2
ACTIVE: subprocess.Popen | None = None


def interrupt_transport() -> None:
    """Interrupt even if the active child exits while pressure is checked."""
    active = ACTIVE
    try:
        if active is not None and active.poll() is None:
            active.terminate()
    finally:
        os.kill(os.getpid(), signal.SIGINT)


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


@contextlib.contextmanager
def shared_node_guard():
    """Hold the required lock and monitor pressure throughout the short transfer."""
    with open("/tmp/exe-codex-local-heavy.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = available_memory()
        if (
            available < 2.5 * GIB
            or available - ESTIMATED_WORKING_SET < 2 * GIB
            or os.getloadavg()[0] >= 1.5
        ):
            raise RuntimeError("Shared VM lacks the required memory/load headroom")
        os.nice(10)
        subprocess.run(
            ["ionice", "-c", "2", "-n", "7", "-p", str(os.getpid())], check=True
        )
        for key in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
            os.environ[key] = "2"
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[key] = "1"
        stop = threading.Event()

        def monitor():
            while not stop.wait(2):
                try:
                    unsafe = available_memory() < 2 * GIB or os.getloadavg()[0] > 2.5
                except (OSError, ValueError, RuntimeError):
                    unsafe = True
                if unsafe:
                    interrupt_transport()
                    return

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=3)
            if ACTIVE is not None and ACTIVE.poll() is None:
                ACTIVE.kill()
                ACTIVE.wait()


def gcloud(arguments: list[str], output: Path | None = None) -> dict | None:
    global ACTIVE
    with contextlib.ExitStack() as stack:
        stream = stack.enter_context(output.open("wb")) if output else subprocess.PIPE
        ACTIVE = subprocess.Popen(
            ["gcloud", "storage", *arguments], stdout=stream, stderr=subprocess.PIPE
        )
        try:
            stdout, _ = ACTIVE.communicate(timeout=300)
            if ACTIVE.returncode:
                raise RuntimeError(f"gcloud storage {arguments[0]} failed")
            return json.loads(stdout) if output is None else None
        finally:
            if ACTIVE.poll() is None:
                ACTIVE.kill()
                ACTIVE.wait()
            ACTIVE = None


def digests(stream) -> dict:
    md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
    while chunk := stream.read(1024**2):
        md5.update(chunk)
        sha256.update(chunk)
        size += len(chunk)
    return {
        "size": size,
        "md5": base64.b64encode(md5.digest()).decode(),
        "sha256": sha256.hexdigest(),
    }


def matches(actual: dict, source: dict) -> None:
    if actual["size"] != int(source["size"]) or actual["md5"] != source["md5_hash"]:
        raise RuntimeError("Checkpoint bytes differ from the pinned GCS object")


def existing_object(client, name: str) -> dict | None:
    try:
        response = client.get_object(Bucket=BUCKET, Key=PREFIX + name)
    except ClientError as exc:
        if str(exc.response["Error"]["Code"]) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    with contextlib.closing(response["Body"]) as body:
        return digests(body)


def validate_config(config: dict) -> None:
    expected = {
        "model_type": "qwen3",
        "vocab_size": 8,
        "hidden_size": 640,
        "intermediate_size": 2560,
        "num_hidden_layers": 7,
        "num_attention_heads": 5,
        "num_key_value_heads": 5,
        "head_dim": 128,
        "max_position_embeddings": 10240,
        "tie_word_embeddings": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Unexpected checkpoint model configuration: {key}")


def stage(*, apply: bool) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    registry = yaml.safe_load(
        subprocess.check_output(
            ["git", "show", f"{commit}:snakemake/analysis/evals_v2/config/config.yaml"],
            cwd=ROOT,
        )
    )
    (model,) = [entry for entry in registry["models"] if entry["name"] == MODEL]
    source_root = model["gcs_path"].rstrip("/")
    if not source_root.startswith("gs://marin-") or not source_root.endswith(
        "/hf/step-100000"
    ):
        raise RuntimeError("The registry does not select the final GCS export")
    source = {}
    for name in FILES:
        source[name] = gcloud(
            [
                "objects",
                "describe",
                f"{source_root}/{name}",
                "--format=json(name,size,generation,md5_hash,crc32c_hash)",
            ]
        )
        if not source[name].get("generation") or not source[name].get("md5_hash"):
            raise RuntimeError("GCS object lacks its required generation or MD5")
    if sum(int(item["size"]) for item in source.values()) > 200_000_000:
        raise RuntimeError("Checkpoint exceeds this bounded transport's size limit")
    report = {
        "consumer_commit": commit,
        "model": MODEL,
        "source": source_root,
        "destination": f"s3://{BUCKET}/{PREFIX}",
        "source_objects": source,
        "applied": False,
        "verified_objects": {},
    }
    if not apply:
        return report
    client = boto3.client("s3", config=Config(connect_timeout=10, read_timeout=60))
    listed = client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=20)
    existing = {item["Key"][len(PREFIX) :] for item in listed.get("Contents", [])}
    if listed.get("IsTruncated") or existing - set(FILES) - {".snakemake_timestamp"}:
        raise RuntimeError("Unexpected objects occupy the canonical checkpoint prefix")
    with tempfile.TemporaryDirectory(prefix="issue550-final-checkpoint-") as temporary:
        root = Path(temporary)
        # Pin downloads to each described generation, then validate every file
        # before any upload; config/schema failures leave S3 untouched.
        for name in FILES:
            path = root / name
            gcloud(["cat", f"{source_root}/{name}#{source[name]['generation']}"], path)
            with path.open("rb") as stream:
                matches(digests(stream), source[name])
        validate_config(json.loads((root / "config.json").read_text()))
        # Detect any incompatible existing checkpoint before the first write.
        verified = {}
        for name in FILES:
            actual = existing_object(client, name)
            if actual is not None:
                matches(actual, source[name])
            verified[name] = actual
        for name in FILES:
            actual = verified[name]
            if actual is None:
                with (root / name).open("rb") as stream:
                    client.put_object(
                        Bucket=BUCKET,
                        Key=PREFIX + name,
                        Body=stream,
                        ContentMD5=source[name]["md5_hash"],
                        IfNoneMatch="*",
                        Metadata={
                            "source-gcs-generation": str(source[name]["generation"])
                        },
                    )
                actual = existing_object(client, name)
            if actual is None:
                raise RuntimeError("Uploaded checkpoint object is missing")
            matches(actual, source[name])
            report["verified_objects"][name] = actual
        if ".snakemake_timestamp" not in existing:
            client.put_object(
                Bucket=BUCKET,
                Key=PREFIX + ".snakemake_timestamp",
                Body=b"",
                ContentMD5=base64.b64encode(hashlib.md5(b"").digest()).decode(),
                IfNoneMatch="*",
            )
    report["applied"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    start_clock = time.monotonic()
    report = {"model": MODEL, "applied": False, "exit_status": 1}
    try:
        with shared_node_guard():
            report.update(stage(apply=args.apply))
        report["exit_status"] = 0
    except KeyboardInterrupt:
        report.update(exit_status=130, error_type="Interrupted")
    except Exception as exc:  # noqa: BLE001 -- sanitize the outer CLI failure receipt
        # SDK exception messages can contain signed URLs; receipts only retain
        # the exception class and never print credentials or raw stderr.
        report["error_type"] = type(exc).__name__
    finally:
        report.update(
            {
                "started_at": started,
                "finished_at": datetime.now(UTC).isoformat(),
                "seconds": time.monotonic() - start_clock,
                "estimated_working_set_bytes": ESTIMATED_WORKING_SET,
                "peak_rss_upper_bound_bytes": 1024
                * (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                ),
            }
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    raise SystemExit(report["exit_status"])


if __name__ == "__main__":
    main()
