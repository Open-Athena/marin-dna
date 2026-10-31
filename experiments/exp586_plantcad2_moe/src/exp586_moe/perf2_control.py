"""Persist and launch one isolated perf2 job with a verified immutable backup."""

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

from exp586_moe.perf2 import ARTIFACT_ROOT, VARIANTS, PerfConfig, remaining_seconds

ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / ".agents/skills/run-training-sweep-cw/scripts/persistence.py"
STATE_DIR = ROOT / "scratch/exp586_perf2"
DATABASE = STATE_DIR / "perf2.sqlite"


def _mutate(*arguments: str) -> None:
    subprocess.run([sys.executable, str(HELPER), *arguments], check=True)
    snapshot = STATE_DIR / (
        dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex
        + ".sqlite"
    )
    subprocess.run(
        [sys.executable, str(HELPER), "backup", str(DATABASE), str(snapshot)],
        check=True,
    )
    fs = fsspec.filesystem("s3")
    target = f"{ARTIFACT_ROOT}/operations/{snapshot.name}".removeprefix("s3://")
    payload = snapshot.read_bytes()
    fs.pipe(target, payload)
    saved = fs.cat(target)
    if (
        len(saved) != len(payload)
        or hashlib.sha256(saved).digest() != hashlib.sha256(payload).digest()
    ):
        raise ValueError("Immutable ledger backup did not verify")
    print(f"Verified immutable backup s3://{target}", flush=True)


def configure_storage() -> None:
    for variable, secret in (
        ("CW_KEY_ID", "cw-object-storage-key-id"),
        ("CW_KEY_SECRET", "cw-object-storage-key-secret"),
    ):
        os.environ[variable] = subprocess.check_output(
            [
                "gcloud",
                "secrets",
                "versions",
                "access",
                "latest",
                "--project=hai-gcp-models",
                f"--secret={secret}",
            ],
            text=True,
        ).strip()
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CW_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CW_KEY_SECRET"]
    os.environ["AWS_REGION"] = "US-EAST-02A"
    os.environ["AWS_ENDPOINT_URL_S3"] = "https://cwobject.com"
    configure_coreweave_s3()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--cluster", choices=["cw-rno2a", "cw-us-east-02a"], required=True
    )
    parser.add_argument("--attempt", default="a1")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    config = PerfConfig(**vars(parser.parse_args()))
    if remaining_seconds() < 300:
        raise ValueError("Deadline does not allow another profiling job")
    configure_storage()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not DATABASE.exists():
        _mutate("init", str(DATABASE))
    _mutate(
        "trial-add",
        str(DATABASE),
        "--trial-id",
        config.run_id,
        "--env-json",
        json.dumps(dataclasses.asdict(config), sort_keys=True),
        "--wandb-run-id",
        config.run_id,
        "--wandb-url",
        f"https://forge.coreweave.com/wandb/eric-czech/marin/runs/{config.run_id}",
        "--checkpoint-root",
        config.artifact_root,
    )
    job_name = config.job_name
    command = [
        "uv",
        "run",
        "--locked",
        "--extra",
        "cpu",
        "iris",
        "--cluster",
        "marin",
        "job",
        "run",
        "--no-wait",
        "--target-cluster",
        config.cluster,
        "--priority",
        "batch",
        "--user",
        "eczech",
        "--job-name",
        job_name,
        "--cpu",
        "2",
        "--memory",
        "6GB",
        "--disk",
        "32GB",
        "--enable-extra-resources",
        "--timeout",
        str(min(3600, remaining_seconds())),
        "--extra",
        "cpu",
        "-e",
        "MARIN_PREFIX",
        "s3://marin-us-east-02a/marin",
    ]
    redacted = command.copy()
    for name in (
        "WANDB_API_KEY",
        "CW_KEY_ID",
        "CW_KEY_SECRET",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        command.extend(["-e", name, os.environ[name]])
        redacted.extend(["-e", name, "<forwarded>"])
    entry = [
        "--",
        "python",
        "-m",
        "exp586_moe.perf2",
        "--variant",
        config.variant,
        "--cluster",
        config.cluster,
        "--attempt",
        config.attempt,
        "--updates",
        str(config.updates),
    ]
    if config.verification:
        entry.append("--verification")
    if config.benchmark:
        entry.append("--benchmark")
    command.extend(entry)
    redacted.extend(entry)
    _mutate(
        "dispatch-intent",
        str(DATABASE),
        "--trial-id",
        config.run_id,
        "--dispatch-id",
        job_name,
        "--iris-job-id",
        f"/eczech/{job_name}",
        "--cluster",
        config.cluster,
        "--gpu-variant",
        "H100",
        "--nodes",
        "1",
        "--gpus",
        "8",
        "--command-redacted",
        shlex.join(redacted),
    )
    print(shlex.join(redacted), flush=True)
    result = subprocess.run(
        command, cwd=ROOT / "experiments/exp586_plantcad2_moe", check=False
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    _mutate(
        "dispatch-confirm",
        str(DATABASE),
        "--dispatch-id",
        job_name,
    )


if __name__ == "__main__":
    main()
