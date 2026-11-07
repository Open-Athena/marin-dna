"""Isolated CUDA-event forward measurements for the exp586 Torch model.

Tokenization, input copies, output copies, compilation and warmup are excluded.
Each process has one configuration so freezing is selected before Torch imports.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


def latency_summary(milliseconds: list[float], batch_size: int) -> dict[str, float]:
    """Convert CUDA-event milliseconds to dense full-window forward throughput."""
    if (
        batch_size < 1
        or not milliseconds
        or any(not np.isfinite(x) or x <= 0 for x in milliseconds)
    ):
        raise ValueError("Positive latencies and batch size are required")
    values = np.asarray(milliseconds, dtype=np.float64)
    median = float(np.median(values))
    return {
        "median_forward_ms": median,
        "p10_forward_ms": float(np.quantile(values, 0.1)),
        "p90_forward_ms": float(np.quantile(values, 0.9)),
        "forward_tokens_per_second": batch_size * 8192 * 1000 / median,
        "forward_sequences_per_second": batch_size * 1000 / median,
    }


def fixture_batch(sequences: list[str], batch_size: int, offset: int) -> list[str]:
    """Rotate complete unique windows without padding or duplicating batch rows."""
    if not 1 <= batch_size <= len(sequences):
        raise ValueError("Batch size must fit the unique fixture")
    return [sequences[(offset + row) % len(sequences)] for row in range(batch_size)]


def validation_batches(
    validation: list[str], sequences: list[str], batch_size: int
) -> Iterator[tuple[list[str], int]]:
    """Keep validation shapes identical to timing without adding scored rows."""
    if not 1 <= batch_size <= len(sequences):
        raise ValueError("Batch size must fit the unique fixture")
    for offset in range(0, len(validation), batch_size):
        real = validation[offset : offset + batch_size]
        padding = [sequence for sequence in sequences if sequence not in real]
        yield real + padding[: batch_size - len(real)], len(real)


def write_result(path: Path, payload: dict[str, Any]) -> None:
    """Publish each trial atomically, including failed/OOM observations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_trial(args: argparse.Namespace) -> dict[str, Any]:
    """Load one model, check fixed-fixture parity, then measure device forwards."""
    # The environment is set by the parent before this process imports Torch.
    import importlib.metadata
    import importlib.util

    import torch

    from exp586_deployment.correctness import acgt_parity
    from exp586_deployment.fixtures import load_fixture, sha256_file
    from exp586_deployment.model import DeploymentModel
    from exp586_deployment.providers import RuntimeConfig
    from exp586_deployment.tokenizer import (
        compact_dna_unigram_tokenizer,
        encode_dna_batch,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment requires H100")
    device = torch.device("cuda:0")
    sequences = load_fixture(args.fixture)
    validation = load_fixture(args.validation_fixture)
    tokenizer = compact_dna_unigram_tokenizer()
    print("STAGE loading-model", flush=True)
    if args.require_no_vllm and importlib.util.find_spec("vllm") is not None:
        raise RuntimeError("The standalone comparison requires vLLM genuinely absent")
    runtime = RuntimeConfig(args.attention_provider, args.fa_version, args.moe_backend)
    model = DeploymentModel(args.model_dir, device, runtime)
    model.prepare_rope(8192, device)
    forward = model

    compiled = torch.compile(
        forward, fullgraph=True, dynamic=False, mode=args.compile_mode
    )

    def encode(batch: list[str]) -> torch.Tensor:
        return torch.from_numpy(encode_dna_batch(tokenizer, batch)).to(device)

    # Always check precisely the same complete real-window fixture.
    # Pad with other real fixture windows so validation cannot leave an unused
    # second compiled graph/CUDA workspace that distorts maximum-fitting batch.
    expected = np.load(args.reference)["logits"]
    if expected.shape != (len(validation), 8192, 4):
        raise ValueError("Reference must cover the complete validation fixture")
    eager_parity = None
    if args.validate_eager:
        eager_parts = []
        with torch.inference_mode():
            for batch, real_rows in validation_batches(
                validation, sequences, args.batch_size
            ):
                ids = encode(batch)
                eager_parts.append(
                    forward(ids)[:real_rows].float().cpu().numpy().copy()
                )
        eager_parity = acgt_parity(np.concatenate(eager_parts), expected, validation)
        del eager_parts, ids
        if not eager_parity["accepted"]:
            return {"status": "eager_parity_failed", "eager_parity": eager_parity}
    actual_parts = []
    print("STAGE compiling-and-validating", flush=True)
    with torch.inference_mode():
        for batch, real_rows in validation_batches(
            validation, sequences, args.batch_size
        ):
            ids = encode(batch)
            actual_parts.append(compiled(ids)[:real_rows].float().cpu().numpy().copy())
    actual = np.concatenate(actual_parts)
    parity = acgt_parity(actual, expected, validation)
    del actual_parts, actual, ids
    if not parity["accepted"]:
        return {
            "status": "parity_failed",
            "parity": parity,
            "eager_parity": eager_parity,
        }
    print("STAGE parity-passed", flush=True)

    # CPU tokenization and H2D complete before recording each start event.
    # No prefetch worker or other process runs tokenizer work during forwards.
    input_batches = args.input_batches or math.ceil(len(sequences) / args.batch_size)
    device_batches = [
        encode(fixture_batch(sequences, args.batch_size, offset * args.batch_size))
        for offset in range(input_batches)
    ]
    torch.cuda.synchronize()
    print("STAGE warmup", flush=True)
    with torch.inference_mode():
        for i in range(args.warmup):
            result = compiled(device_batches[i % len(device_batches)])
            torch.cuda.synchronize()
            if not torch.isfinite(result).all().item():
                raise ValueError("Nonfinite warmup output")
        del result
        torch.cuda.reset_peak_memory_stats()
        timings: list[float] = []
        print("STAGE timing", flush=True)
        # Synchronize every iteration: the interval includes the entire forward,
        # including any dispatch gaps, but no host-output copy or next-batch work.
        for i in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = compiled(device_batches[i % len(device_batches)])
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
            del result
    config = json.loads((args.model_dir / "config.json").read_text())
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "status": "passed",
        "eager_parity": eager_parity,
        "optional_packages": {
            name: importlib.metadata.version(name)
            if importlib.util.find_spec(module) is not None
            else None
            for name, module in [
                ("vllm", "vllm"),
                ("flash-attn", "flash_attn"),
                ("flash-attn-3", "flash_attn_interface"),
            ]
        },
        "vllm_imported": any(
            name == "vllm" or name.startswith("vllm.") for name in sys.modules
        ),
        **latency_summary(timings, args.batch_size),
        "forward_ms": timings,
        "timed_input_batches": input_batches,
        "fixture_windows": len(sequences),
        "parity": parity,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "device_used_gib_after_timing": (total_bytes - free_bytes) / 2**30,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "checkpoint": config.get("exp586_checkpoint"),
        "checkpoint_metadata_digest": config.get("exp586_checkpoint_metadata_digest"),
        "weight_index_sha256": sha256_file(
            args.model_dir / "model.safetensors.index.json"
        ),
        "tokenizer_sha256": sha256_file(args.model_dir / "tokenizer.json"),
        "fixture_sha256": sha256_file(args.fixture),
        "validation_fixture_sha256": sha256_file(args.validation_fixture),
        "reference_sha256": sha256_file(args.reference),
    }


def main() -> int:
    """Run a single self-describing fresh-process trial."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--validation-fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--input-batches",
        type=int,
        default=0,
        help="Zero rotates through the complete fixture",
    )
    parser.add_argument(
        "--attention-provider", choices=["standalone", "vllm"], required=True
    )
    parser.add_argument("--fa-version", type=int, choices=[2, 3], required=True)
    parser.add_argument("--moe-backend", choices=["torch", "vllm"], default="torch")
    parser.add_argument("--require-no-vllm", action="store_true")
    parser.add_argument("--validate-eager", action="store_true")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.iterations < 2
        or args.warmup < 1
        or args.input_batches < 0
    ):
        parser.error("Positive batch/input/warmup and at least two iterations required")
    os.environ.pop("TORCHINDUCTOR_FREEZING", None)
    began = time.time()
    payload: dict[str, Any] = {
        "schema": 2,
        "source_revision": args.source_revision,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(
                {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.glob("*.py"))
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "name": args.name,
        "model_repository": "plantcad/marin-dna-exp586-1.5B-60M-216B",
        "model_revision": "4130f71edcd117a7375af1ac78723d9a9173533f",
        "started_unix": began,
        "metric_scope": "CUDA-event device forward only; no H2D/D2H/tokenization",
        "config": {
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "input_batches": args.input_batches,
            "freezing": os.environ.get("TORCHINDUCTOR_FREEZING", "0"),
            "specialized_topk": False,
            "sconv_backend": "direct",
            "combined_shared_expert_projections": False,
            "precomputed_rope": True,
            "attention_provider": args.attention_provider,
            "flash_attention_version": args.fa_version,
            "moe_backend": args.moe_backend,
            "compile_mode": args.compile_mode,
            "fullgraph": True,
            "dynamic": False,
            "compact_input": "ACGTN",
            "compact_output": "ACGT",
            "packed_vllm_routed_weights": args.moe_backend == "vllm",
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        payload.update(run_trial(args))
    except Exception as error:  # noqa: BLE001 -- persist backend/compiler failures as trial evidence
        message = str(error)
        payload.update(
            status="oom"
            if "out of memory" in message.lower()
            or "outofmemory" in type(error).__name__.lower()
            else "error",
            error_type=type(error).__name__,
            error=message,
            traceback=traceback.format_exc(),
        )
    payload["process_seconds"] = time.time() - began
    try:
        payload["nvidia_smi"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    write_result(args.output, payload)
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k not in {"forward_ms", "traceback"}},
            indent=2,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
