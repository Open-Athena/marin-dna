"""Measure prompt-only vLLM inference on one H100."""

import json
import os
import time
import types
from pathlib import Path
from typing import Literal

import numpy as np
from tokenizers import Tokenizer

from exp586_moe.inference.correctness import acgt_parity
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.ledger import Trial, append_trial
from exp586_moe.inference.tokenizer import encode_dna, reduced_dna_tokenizer

VllmVariant = Literal["full_head", "acgt_head"]
TokenizerVariant = Literal["full", "reduced"]


def _install_capture(worker, base_ids: list[int], variant: VllmVariant) -> bool:
    """Capture all-position logits from real vLLM prompt-prefill forwards."""
    import torch
    from torch.nn import functional

    model = worker.model_runner.model
    original_forward = model.forward
    head = model.lm_head.weight[base_ids].detach()
    model._exp586_captures = []

    def capturing_forward(self, *args, **kwargs):
        hidden = original_forward(*args, **kwargs)
        tensor = hidden[0] if isinstance(hidden, tuple) else hidden
        if isinstance(tensor, torch.Tensor) and tensor.shape[0] >= 8192:
            prompt_tokens = (tensor.shape[0] // 8192) * 8192
            states = tensor[:prompt_tokens].reshape(-1, 8192, tensor.shape[-1])
            if variant == "full_head":
                selected = functional.linear(states, self.lm_head.weight)[..., base_ids]
            else:
                selected = functional.linear(states, head)
            self._exp586_captures.append(selected.float().cpu().numpy())
        return hidden

    model.forward = types.MethodType(capturing_forward, model)
    return True


def _clear_captures(worker) -> int:
    captures = worker.model_runner.model._exp586_captures
    count = len(captures)
    captures.clear()
    return count


def _save_captures(worker, output: str) -> int:
    captures = worker.model_runner.model._exp586_captures
    values = np.concatenate(captures, axis=0)
    np.savez_compressed(output, logits=values)
    return values.shape[0]


def benchmark(
    model_dir: str,
    fixture: str,
    ledger: str,
    output: str,
    *,
    variant: VllmVariant = "full_head",
    name: str = "baseline",
    reference: str,
    tokenizer_variant: TokenizerVariant = "full",
    compile_engine: bool = False,
    gpu_memory_utilization: float = 0.2,
    kv_cache_memory_bytes: int | None = None,
    moe_backend: str = "auto",
    attention_backend: str | None = None,
    flash_attention_version: int | None = None,
    batch_size: int = 1,
    iterations: int = 8,
    validation_fixture: str | None = None,
) -> None:
    """Measure batched 8,192-token prefills and return dense A/C/G/T logits."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # vLLM reserves one generated token to trigger a standard engine prefill. The
    # model never evaluates position 8192; only the requested 8,192-token prompt.
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    from vllm import LLM, SamplingParams

    began_productive = time.perf_counter()
    sequences = load_fixture(Path(fixture))
    validation_sequences = load_fixture(
        Path(validation_fixture) if validation_fixture is not None else Path(fixture)
    )
    if batch_size < 1 or batch_size > len(sequences):
        raise ValueError("Batch size must fit the unique-sequence fixture")
    if iterations < 2:
        raise ValueError("Use at least two timed iterations")
    tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    if tokenizer_variant == "reduced":
        tokenizer = reduced_dna_tokenizer(tokenizer)
    elif tokenizer_variant != "full":
        raise ValueError(f"Unknown tokenizer variant: {tokenizer_variant}")
    base_ids = [tokenizer.token_to_id(base.lower()) for base in "ACGT"]
    if any(token_id is None for token_id in base_ids):
        raise ValueError("Tokenizer lacks isolated A/C/G/T tokens")
    resolved_base_ids = [int(token_id) for token_id in base_ids]

    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=not compile_engine,
        max_model_len=8193,
        max_num_batched_tokens=batch_size * 8193,
        max_num_seqs=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        disable_log_stats=True,
        moe_backend=moe_backend,
        attention_config=(
            {
                **({"backend": attention_backend} if attention_backend else {}),
                **(
                    {"flash_attn_version": flash_attention_version}
                    if flash_attention_version
                    else {}
                ),
            }
            if attention_backend or flash_attention_version
            else None
        ),
    )
    installed = llm.llm_engine.collective_rpc(
        _install_capture, args=(resolved_base_ids, variant)
    )
    if installed != [True]:
        raise RuntimeError(f"Capture hook was not installed: {installed}")
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)

    def batch_at(index: int) -> list[str]:
        start = (index * batch_size) % len(sequences)
        return [
            sequences[(start + offset) % len(sequences)] for offset in range(batch_size)
        ]

    def run(batch: list[str]) -> None:
        prompts = [
            {"prompt_token_ids": encode_dna(tokenizer, sequence).tolist()}
            for sequence in batch
        ]
        llm.generate(
            prompts,
            sampling,
            use_tqdm=False,
        )

    run(batch_at(0))
    llm.llm_engine.collective_rpc(_clear_captures)

    latencies: list[float] = []
    for index in range(iterations):
        began = time.perf_counter()
        run(batch_at(index))
        latencies.append(time.perf_counter() - began)

    # Discard timed captures and validate every batch size on the same complete
    # fixture. This keeps throughput batch size separate from sample coverage.
    llm.llm_engine.collective_rpc(_clear_captures)
    validation_batch_size = min(batch_size, len(validation_sequences))
    for start in range(0, len(validation_sequences), validation_batch_size):
        run(validation_sequences[start : start + validation_batch_size])

    capture_path = str(Path(output).with_suffix(".reference.npz"))
    captured = llm.llm_engine.collective_rpc(_save_captures, args=(capture_path,))
    expected_captures = len(validation_sequences)
    if captured != [expected_captures]:
        raise RuntimeError(f"Expected {expected_captures} captures, got {captured}")
    actual = np.load(capture_path)["logits"]
    expected = np.load(reference)["logits"]
    if expected.shape[0] != len(validation_sequences):
        raise ValueError("Correctness reference must cover the complete fixture")
    parity = acgt_parity(actual, expected, validation_sequences)

    elapsed = time.perf_counter() - began_productive
    median = float(np.median(latencies))
    sequences_per_second = batch_size / median
    tokens_per_sequence = 8192
    payload = {
        "backend": "vllm",
        "variant": variant,
        "name": name,
        "batch_size": batch_size,
        "sequence_length": tokens_per_sequence,
        "iterations": iterations,
        "latency_seconds": latencies,
        "median_latency_seconds": median,
        "tokens_per_sequence": tokens_per_sequence,
        "input_tokens_per_second": sequences_per_second * tokens_per_sequence,
        "sequences_per_second": sequences_per_second,
        "productive_seconds": elapsed,
        "parity": parity,
        "accepted": bool(parity["accepted"]),
        "compiled": compile_engine,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "tokenizer_variant": tokenizer_variant,
        "moe_backend": moe_backend,
        "attention_backend": attention_backend,
        "flash_attention_version": flash_attention_version,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    append_trial(
        Path(ledger),
        Trial.now(
            backend="vllm",
            name=name,
            productive_seconds=elapsed,
            sequences_per_second=sequences_per_second,
            median_latency_seconds=median,
            accepted=bool(payload["accepted"]),
            correctness_max_abs_error=float(parity["raw_max_abs_error"]),
            notes=(
                f"{variant}; compiled={compile_engine}; tokenizer={tokenizer_variant}; "
                f"gpu_memory_utilization={gpu_memory_utilization}; "
                f"kv_cache_memory_bytes={kv_cache_memory_bytes}; "
                f"moe={moe_backend}; attention={attention_backend or 'auto'}; "
                f"flash_attention_version={flash_attention_version or 'auto'}; "
                f"batch={batch_size}"
            ),
            tokens_per_sequence=tokens_per_sequence,
            batch_size=batch_size,
        ),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--variant", choices=("full_head", "acgt_head"), default="full_head"
    )
    parser.add_argument("--name", default="baseline")
    parser.add_argument(
        "--tokenizer-variant", choices=("full", "reduced"), default="full"
    )
    parser.add_argument("--compile-engine", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--validation-fixture")
    parser.add_argument(
        "--moe-backend",
        choices=(
            "auto",
            "triton",
            "batched_triton",
            "flashinfer_trtllm",
            "flashinfer_cutlass",
        ),
        default="auto",
    )
    parser.add_argument(
        "--attention-backend",
        choices=(
            "FLASH_ATTN",
            "TRITON_ATTN",
            "TORCH_SDPA",
            "FLASHINFER",
            "FLEX_ATTENTION",
        ),
    )
    parser.add_argument("--flash-attention-version", type=int, choices=(2, 3, 4))
    arguments = parser.parse_args()
    benchmark(
        arguments.model_dir,
        arguments.fixture,
        arguments.ledger,
        arguments.output,
        variant=arguments.variant,
        name=arguments.name,
        reference=arguments.reference,
        tokenizer_variant=arguments.tokenizer_variant,
        compile_engine=arguments.compile_engine,
        gpu_memory_utilization=arguments.gpu_memory_utilization,
        kv_cache_memory_bytes=arguments.kv_cache_memory_bytes,
        moe_backend=arguments.moe_backend,
        attention_backend=arguments.attention_backend,
        flash_attention_version=arguments.flash_attention_version,
        batch_size=arguments.batch_size,
        iterations=arguments.iterations,
        validation_fixture=arguments.validation_fixture,
    )
