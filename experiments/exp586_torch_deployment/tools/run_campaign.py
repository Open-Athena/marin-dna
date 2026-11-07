"""Run an explicit trial catalog serially, preserving every process result."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def run(plan_path: Path, root: Path, output: Path, prefix: str, revision: str) -> None:
    """Use separately installed environments; no resource or recovery decisions."""
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url="http://cwlota.com",
        region_name="US-EAST-02A",
        config=Config(
            s3={"addressing_style": "virtual"},
            retries={"total_max_attempts": 1},
            connect_timeout=30,
            read_timeout=120,
        ),
    )
    bucket, key_prefix = prefix.removeprefix("s3://").strip("/").split("/", 1)

    def upload(path: Path) -> None:
        s3.upload_file(str(path), bucket, key_prefix + "/" + path.name)

    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for spec in plan:
            name = spec["name"]
            result_path = output / (name + ".json")
            if result_path.exists():
                recorded = json.loads(result_path.read_text())
                assert recorded["status"] == "passed", name
                assert recorded["source_revision"] == revision, name
                assert recorded["campaign_spec"] == spec, name
                print("ALREADY_RECORDED", name, flush=True)
                continue
            environment = "vllm" if spec["use_vllm"] else "no-vllm"
            python = root / environment / "bin/python"
            env = os.environ.copy()
            env.pop("TORCHINDUCTOR_FREEZING", None)
            compile_identity = {
                key: value
                for key, value in spec.items()
                if key not in {"name", "iterations", "warmup", "validate_eager"}
            }
            cache_key = hashlib.sha256(
                json.dumps(compile_identity, sort_keys=True).encode()
            ).hexdigest()[:16]
            env.update(
                CUDA_VISIBLE_DEVICES="0",
                OMP_NUM_THREADS="1",
                TORCHINDUCTOR_COMPILE_THREADS="1",
                TORCHINDUCTOR_CACHE_DIR=str(
                    root / "compiler-cache" / revision / cache_key
                ),
            )
            command = [
                str(python),
                "-m",
                "exp586_deployment.benchmark",
                "--model-dir",
                "/app/forward-artifacts/hf-bf16",
                "--fixture",
                "/app/forward-artifacts/benchmark-fixture-512.json",
                "--validation-fixture",
                "/app/forward-artifacts/benchmark-fixture-32.json",
                "--reference",
                "/app/forward-artifacts/jax-reference32-latest.reference.npz",
                "--output",
                str(result_path),
                "--name",
                name,
                "--batch-size",
                str(spec["batch_size"]),
                "--iterations",
                str(spec["iterations"]),
                "--warmup",
                str(spec["warmup"]),
                "--model-class",
                spec.get("model_class", "ordinary"),
                "--fa-version",
                str(spec["fa"]),
                "--compile-mode",
                spec.get("compile_mode", "default"),
                "--dynamic",
                spec.get("dynamic", "auto"),
                "--source-revision",
                revision,
            ]
            if environment == "no-vllm":
                command.append("--require-no-vllm")
            else:
                command.append("--use-vllm")
            if spec.get("freezing"):
                command.append("--freezing")
            if spec.get("validate_eager"):
                command.append("--validate-eager")
            print("START", name, json.dumps(spec), flush=True)
            log = output / (name + ".log")
            began = time.time()
            with log.open("w") as stream:
                try:
                    process = subprocess.run(
                        command,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=900,
                        check=False,
                    )
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    returncode = -9
            if not result_path.exists():
                result_path.write_text(
                    json.dumps(
                        {
                            "name": name,
                            "status": "process_failed",
                            "returncode": returncode,
                            "spec": spec,
                            "process_seconds": time.time() - began,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            result = json.loads(result_path.read_text())
            result["campaign_spec"] = spec
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            for artifact in [result_path, log]:
                upload(artifact)
            print(
                "DONE",
                name,
                result["status"],
                result.get("forward_tokens_per_second"),
                flush=True,
            )
            if result["status"] != "passed":
                raise RuntimeError(
                    f"{name}: {result['status']}; see saved result and log"
                )
        completion = {
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "source_revision": revision,
            "completed_unix": time.time(),
            "trials": [spec["name"] for spec in plan],
        }
        path = output / "campaign.complete.json"
        path.write_text(json.dumps(completion, indent=2) + "\n")
        upload(path)
        print("CAMPAIGN_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    run(args.plan, args.root, args.output, args.prefix, args.revision)
