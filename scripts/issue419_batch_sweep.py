"""Benchmark m5.1 genome-logo inference batch sizes for issue #419."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

from marin_dna.pipelines.chinchilla_logo import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    score_window_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--sample-windows", type=int, default=16_384)
    parser.add_argument("--windows-per-chunk", type=int, default=4_096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-hourly-cost", type=float, required=True)
    parser.add_argument("--model-repository", default=MODEL_REPOSITORY)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    return parser.parse_args()


def summarize_runtime(
    runtime: dict[str, Any], gpu_hourly_cost: float
) -> dict[str, Any]:
    per_shard = runtime["per_shard"]
    assert len(per_shard) >= 2, "need a warm-up shard and a steady-state shard"
    warmup = per_shard[0]
    steady = per_shard[1:]
    steady_windows = sum(int(row["window_count"]) for row in steady)
    steady_bases = sum(int(row["scored_base_count"]) for row in steady)
    steady_seconds = sum(float(row["inference_seconds"]) for row in steady)
    assert steady_windows > 0 and steady_bases > 0 and steady_seconds > 0
    bases_per_second = steady_bases / steady_seconds
    return {
        **runtime,
        "warmup_chunk_seconds": float(warmup["inference_seconds"]),
        "steady_state_window_count": steady_windows,
        "steady_state_scored_base_count": steady_bases,
        "steady_state_inference_seconds": steady_seconds,
        "steady_state_windows_per_second": steady_windows / steady_seconds,
        "steady_state_bases_per_second": bases_per_second,
        "gpu_hourly_cost_usd": gpu_hourly_cost,
        "steady_state_usd_per_billion_scored_bases": (
            1_000_000_000 / bases_per_second / 3_600 * gpu_hourly_cost
        ),
    }


def main() -> None:
    args = parse_args()
    assert args.sample_windows > args.windows_per_chunk > 0
    assert args.sample_windows % args.windows_per_chunk == 0
    assert args.gpu_hourly_cost > 0
    assert args.batch_sizes and all(batch_size > 0 for batch_size in args.batch_sizes)
    assert len(args.batch_sizes) == len(set(args.batch_sizes))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_plan = args.output_dir / "sample.parquet"
    plan = pl.read_parquet(args.plan).head(args.sample_windows)
    assert plan.height == args.sample_windows, (
        f"requested {args.sample_windows} windows, plan has only {plan.height}"
    )
    plan.write_parquet(sample_plan)

    results: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        batch_dir = args.output_dir / f"batch-{batch_size}"
        runtime = score_window_plan(
            sample_plan,
            args.genome,
            batch_dir / "shards",
            batch_dir / "runtime.json",
            batch_dir / "done.json",
            model_repository=args.model_repository,
            model_revision=args.model_revision,
            windows_per_chunk=args.windows_per_chunk,
            batch_size=batch_size,
            num_workers=args.num_workers,
            torch_compile=True,
            bf16_full_eval=True,
        )
        results.append(summarize_runtime(runtime, args.gpu_hourly_cost))
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "sample_plan": str(sample_plan),
        "sample_windows": args.sample_windows,
        "windows_per_chunk": args.windows_per_chunk,
        "results": results,
        "recommended_batch_size": max(
            results, key=lambda result: result["steady_state_bases_per_second"]
        )["batch_size"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
