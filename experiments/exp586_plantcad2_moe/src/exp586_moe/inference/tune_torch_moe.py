"""Tune the vLLM Triton MoE kernel for the single-H100 scoring workload."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 16,
    "SPLIT_K": 1,
    "num_warps": 8,
    "num_stages": 3,
}

SEARCH_VALUES = {
    "BLOCK_SIZE_M": (32, 64, 128, 256),
    "BLOCK_SIZE_N": (32, 64, 128, 256),
    "BLOCK_SIZE_K": (64, 128, 256),
    "GROUP_SIZE_M": (1, 16, 32, 64),
    "num_warps": (4, 8),
    "num_stages": (2, 3, 4, 5),
}


def tune(
    output: str,
    *,
    batch_size: int = 128,
    iterations: int = 7,
    coordinate_passes: int = 2,
) -> None:
    """Run coordinate descent over kernel tiles at the real routed-token shape."""
    if batch_size < 1 or iterations < 2 or coordinate_passes < 1:
        raise ValueError(
            "batch_size, iterations and coordinate_passes must be positive"
        )
    began = time.perf_counter()
    torch.manual_seed(0)
    torch.set_default_device("cuda:0")
    tokens = batch_size * 8192
    experts = 384
    hidden = 384
    intermediate = 384
    top_k = 8
    routed = torch.randn(tokens, hidden, dtype=torch.bfloat16) * 0.02
    w1 = torch.randn(experts, 2 * intermediate, hidden, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16) * 0.02
    rows = torch.arange(tokens, dtype=torch.int64)[:, None]
    slots = torch.arange(top_k, dtype=torch.int64)[None, :]
    selected = ((rows * top_k + slots) % experts).to(torch.int32)
    combine = torch.rand(tokens, top_k, dtype=torch.float32)
    combine = combine / combine.sum(dim=-1, keepdim=True)
    trials: list[dict[str, object]] = []
    seen: set[str] = set()

    def evaluate(config: dict[str, int]) -> float:
        key = json.dumps(config, sort_keys=True)
        if key in seen:
            prior = next(item for item in trials if item["key"] == key)
            return float(prior["median_ms"])
        seen.add(key)
        try:
            with torch.inference_mode(), override_config(config):
                for _ in range(2):
                    result = fused_experts(
                        routed,
                        w1,
                        w2,
                        combine,
                        selected,
                        global_num_experts=experts,
                    )
                torch.cuda.synchronize()
                latencies = []
                for _ in range(iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = fused_experts(
                        routed,
                        w1,
                        w2,
                        combine,
                        selected,
                        global_num_experts=experts,
                    )
                    end.record()
                    end.synchronize()
                    latencies.append(start.elapsed_time(end))
                del result
            median = float(np.median(latencies))
            trials.append(
                {
                    "key": key,
                    "config": dict(config),
                    "median_ms": median,
                    "latency_ms": latencies,
                }
            )
            print(f"{median:.3f} ms {config}", flush=True)
            return median
        # Triton reports invalid tile shapes through several backend-specific
        # exception classes. A tuner must record and skip all of them.
        except Exception as error:  # noqa: BLE001
            torch.cuda.empty_cache()
            trials.append(
                {
                    "key": key,
                    "config": dict(config),
                    "median_ms": float("inf"),
                    "error": str(error),
                }
            )
            print(f"failed {config}: {error}", flush=True)
            return float("inf")

    current = dict(DEFAULT_CONFIG)
    best_ms = evaluate(current)
    for _ in range(coordinate_passes):
        changed = False
        for field, values in SEARCH_VALUES.items():
            candidates = []
            for value in values:
                candidate = dict(current)
                candidate[field] = value
                candidates.append((evaluate(candidate), candidate))
            candidate_ms, candidate = min(candidates, key=lambda item: item[0])
            if candidate_ms < best_ms:
                current = candidate
                best_ms = candidate_ms
                changed = True
        if not changed:
            break

    payload = {
        "batch_size": batch_size,
        "tokens": tokens,
        "shape": {
            "experts": experts,
            "hidden": hidden,
            "intermediate": intermediate,
            "top_k": top_k,
        },
        "best_config": current,
        "best_median_ms": best_ms,
        "productive_seconds": time.perf_counter() - began,
        "trials": trials,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(tune)
