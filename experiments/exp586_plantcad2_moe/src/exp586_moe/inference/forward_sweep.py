"""Execute an explicit forward-benchmark trial list serially on one GPU."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_plan(plan: Path, root: Path, output: Path, results_prefix: str | None) -> None:
    """Run each declared trial in a fresh process and preserve failures and logs."""
    output.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = json.loads(plan.read_text())
    for spec in specs:
        name = spec["name"]
        destination = output / f"{name}.json"
        if destination.exists():
            print(f"already recorded {name}", flush=True)
            continue
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="0",
            TORCHINDUCTOR_FREEZING=str(int(spec.get("freezing", True))),
            TOKENIZERS_PARALLELISM="true",
        )
        command = [
            sys.executable,
            "-m",
            "exp586_moe.inference.forward_torch",
            "--model-dir",
            str(root / "hf-bf16"),
            "--fixture",
            str(root / "benchmark-fixture-512.json"),
            "--validation-fixture",
            str(root / "benchmark-fixture-32.json"),
            "--reference",
            str(root / "jax-reference32-latest.reference.npz"),
            "--output",
            str(destination),
            "--name",
            name,
            "--batch-size",
            str(spec["batch_size"]),
            "--iterations",
            str(spec.get("iterations", 24)),
            "--warmup",
            str(spec.get("warmup", 5)),
            "--sconv-backend",
            spec.get("sconv_backend", "direct"),
        ]
        if spec.get("specialized_topk", False):
            command.append("--specialized-topk")
        if not spec.get("combined_shared", True):
            command.append("--no-combined-shared")
        log = output / f"{name}.log"
        print(f"START {name} {json.dumps(spec)}", flush=True)
        began = time.time()
        with log.open("w") as stream:
            try:
                process = subprocess.run(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=spec.get("timeout", 900),
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = -9
        if not destination.exists():
            destination.write_text(
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
        result = json.loads(destination.read_text())
        print(
            "DONE "
            + json.dumps(
                {
                    "name": name,
                    "status": result["status"],
                    "tokens_per_second": result.get("forward_tokens_per_second"),
                    "ms": result.get("median_forward_ms"),
                    "process_seconds": result.get("process_seconds"),
                }
            ),
            flush=True,
        )
        if results_prefix:
            import fsspec
            from rigging.filesystem.s3_compat import configure_coreweave_s3

            configure_coreweave_s3()
            fs = fsspec.filesystem("s3")
            for artifact in [destination, log]:
                fs.put_file(
                    str(artifact), results_prefix.rstrip("/") + "/" + artifact.name
                )


def main() -> None:
    """Execute only the configurations supplied by the experiment operator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-prefix")
    args = parser.parse_args()
    run_plan(args.plan, args.root, args.output, args.results_prefix)


if __name__ == "__main__":
    main()
