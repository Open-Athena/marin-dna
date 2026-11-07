"""Run an explicit six-variant catalog serially, preserving every process result."""

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
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for spec in plan:
            name = spec["name"]
            result_path = output / (name + ".json")
            if result_path.exists():
                assert json.loads(result_path.read_text())["status"] == "passed", name
                print("ALREADY_RECORDED", name, flush=True)
                continue
            environment = "no-vllm" if spec["provider"] == "standalone" else "vllm"
            python = root / environment / "bin/python"
            env = os.environ.copy()
            env.pop("TORCHINDUCTOR_FREEZING", None)
            env.update(
                CUDA_VISIBLE_DEVICES="0",
                OMP_NUM_THREADS="1",
                TORCHINDUCTOR_COMPILE_THREADS="1",
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
                "--attention-provider",
                spec["provider"],
                "--fa-version",
                str(spec["fa"]),
                "--moe-backend",
                spec["moe"],
                "--source-revision",
                revision,
            ]
            if environment == "no-vllm":
                command.append("--require-no-vllm")
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
            for artifact in [result_path, log]:
                fs.put_file(str(artifact), prefix.rstrip("/") + "/" + artifact.name)
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
        fs.put_file(str(path), prefix.rstrip("/") + "/" + path.name)
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
