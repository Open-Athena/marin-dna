"""Measure prompt-only vLLM inference on one H100."""

import json
import os
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from tokenizers import Tokenizer

from exp586_moe.inference.correctness import acgt_parity
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.ledger import Trial, append_trial
from exp586_moe.inference.results_store import upload_trial_artifacts
from exp586_moe.inference.tokenizer import (
    encode_dna,
    encode_dna_batch_lists,
    reduced_dna_tokenizer,
    reduced_dna_unigram_tokenizer,
)

VllmVariant = Literal["full_head", "acgt_head"]
TokenizerVariant = Literal["full", "reduced", "reduced_unigram"]


def _install_capture(
    worker,
    base_ids: list[int],
    variant: VllmVariant,
    output_transfer_dtype: Literal["float32", "bfloat16"],
    minimal_sampling_head: bool,
    specialized_topk: bool = False,
    direct_sconv: bool = False,
    moe_config: dict[str, int] | None = None,
    async_output_transfer: bool = False,
    separate_output_transfer_stream: bool = False,
    deferred_output_transfer: bool = False,
    preallocated_output_transfer: bool = False,
    capture_batch_size: int | None = None,
) -> bool:
    """Capture all-position logits from real vLLM prompt-prefill forwards."""
    import torch
    from torch.nn import functional

    model = worker.model_runner.model
    original_forward = model.forward
    head = model.lm_head.weight[base_ids].detach()
    model._exp586_captures = []
    model._exp586_copy_stream = (
        torch.cuda.Stream(device=worker.model_runner.device)
        if separate_output_transfer_stream
        else None
    )
    model._exp586_capture_offset = 0
    model._exp586_output_buffer = None
    if preallocated_output_transfer:
        if capture_batch_size is None or capture_batch_size < 1:
            raise ValueError("Preallocated transfer requires a positive batch size")
        transfer_dtype = (
            torch.float32 if output_transfer_dtype == "float32" else torch.bfloat16
        )
        model._exp586_output_buffer = torch.empty(
            (capture_batch_size, 8192, len(base_ids)),
            dtype=transfer_dtype,
            device="cpu",
            pin_memory=True,
        )

    if specialized_topk:
        from vllm.model_executor.models.grugmoe import GrugMoeRouter

        from exp586_moe.inference.torch_topk import router_top_k

        def compute_routing(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            del hidden_states, input_ids
            selected = router_top_k(
                router_logits.float() + self.bias.float(), self.top_k + 1
            )[:, : self.top_k]
            weights = torch.sigmoid(
                torch.gather(router_logits.float(), dim=-1, index=selected)
            )
            weights = weights * (2.5 / (weights.sum(dim=-1, keepdim=True) + 1e-9))
            return weights, selected.to(
                torch.int32 if indices_type is None else indices_type
            )

        GrugMoeRouter._compute_routing = compute_routing

    if direct_sconv:
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.models.grugmoe import GrugMoeShortConv

        from exp586_moe.inference.torch_sconv import short_conv_prefill_out

        original_sconv = GrugMoeShortConv.forward_impl

        def direct_sconv_prefill(self, hidden_states, output):
            metadata_by_layer = get_forward_context().attn_metadata
            metadata = (
                metadata_by_layer.get(self.prefix)
                if isinstance(metadata_by_layer, dict)
                else None
            )
            if (
                metadata is None
                or metadata.num_decode_tokens != 0
                or metadata.num_prefills == 0
                or metadata.num_prefill_tokens != metadata.num_prefills * 8192
            ):
                return original_sconv(self, hidden_states, output)
            count = metadata.num_prefill_tokens
            short_conv_prefill_out(hidden_states, self.weight, output, count)

        GrugMoeShortConv.forward_impl = direct_sconv_prefill

    def capturing_forward(self, *args, **kwargs):
        if moe_config is not None:
            from vllm.model_executor.layers.fused_moe import override_config

            config_context = override_config(moe_config)
        else:
            config_context = nullcontext()
        with config_context:
            hidden = original_forward(*args, **kwargs)
        tensor = hidden[0] if isinstance(hidden, tuple) else hidden
        if isinstance(tensor, torch.Tensor) and tensor.shape[0] >= 8192:
            prompt_tokens = (tensor.shape[0] // 8192) * 8192
            states = tensor[:prompt_tokens].reshape(-1, 8192, tensor.shape[-1])
            if variant == "full_head":
                selected = functional.linear(states, self.lm_head.weight)[..., base_ids]
            else:
                selected = functional.linear(states, head)
            if deferred_output_transfer:
                captured = (
                    selected.float()
                    if output_transfer_dtype == "float32"
                    else selected.to(torch.bfloat16)
                )
            elif preallocated_output_transfer:
                start = self._exp586_capture_offset
                stop = start + selected.shape[0]
                if stop > self._exp586_output_buffer.shape[0]:
                    raise RuntimeError("Captured logits exceed the preallocated batch")
                captured = self._exp586_output_buffer[start:stop]
                captured.copy_(selected, non_blocking=True)
                self._exp586_capture_offset = stop
            elif async_output_transfer or separate_output_transfer_stream:
                transfer_dtype = (
                    torch.float32
                    if output_transfer_dtype == "float32"
                    else torch.bfloat16
                )
                captured = torch.empty(
                    selected.shape,
                    dtype=transfer_dtype,
                    device="cpu",
                    pin_memory=True,
                )
                if separate_output_transfer_stream:
                    copy_stream = self._exp586_copy_stream
                    copy_stream.wait_stream(torch.cuda.current_stream(selected.device))
                    selected.record_stream(copy_stream)
                    with torch.cuda.stream(copy_stream):
                        captured.copy_(selected, non_blocking=True)
                else:
                    captured.copy_(selected, non_blocking=True)
            elif output_transfer_dtype == "float32":
                captured = selected.float().cpu().numpy()
            else:
                captured = selected.to(torch.bfloat16).cpu().float().numpy()
            self._exp586_captures.append(captured)
        return hidden

    model.forward = types.MethodType(capturing_forward, model)
    if minimal_sampling_head:

        def one_token_logits(self, hidden_states):
            # LLM.generate needs one valid sampled token to finish each request,
            # but this benchmark consumes only the all-position A/C/G/T capture
            # above. A one-column tensor preserves that engine contract without
            # applying the unused 128,256-column language head to final states.
            return torch.zeros(
                (hidden_states.shape[0], 1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        model.compute_logits = types.MethodType(one_token_logits, model)
    return True


def _clear_captures(worker) -> int:
    model = worker.model_runner.model
    captures = model._exp586_captures
    count = len(captures)
    captures.clear()
    model._exp586_capture_offset = 0
    return count


def _reset_peak_memory(worker) -> bool:
    import torch

    torch.cuda.reset_peak_memory_stats(worker.model_runner.device)
    return True


def _finish_async_captures(worker, clear: bool) -> int:
    """Wait once per engine batch after overlapping chunk-level D2H copies."""
    import torch

    model = worker.model_runner.model
    copy_stream = model._exp586_copy_stream
    if copy_stream is None:
        torch.cuda.synchronize(worker.model_runner.device)
    else:
        copy_stream.synchronize()
    captures = model._exp586_captures
    count = len(captures)
    if clear:
        captures.clear()
        model._exp586_capture_offset = 0
    return count


def _finish_deferred_captures(worker, clear: bool) -> int:
    """Combine device logits and perform one host transfer per engine batch."""
    import torch

    model = worker.model_runner.model
    captures = model._exp586_captures
    count = len(captures)
    if not captures:
        return 0
    merged = torch.cat(captures, dim=0)
    host = torch.empty(
        merged.shape,
        dtype=merged.dtype,
        device="cpu",
        pin_memory=True,
    )
    host.copy_(merged, non_blocking=True)
    torch.cuda.synchronize(worker.model_runner.device)
    captures.clear()
    if not clear:
        captures.append(host)
    return count


def _runtime_info(worker) -> dict[str, str | float]:
    import torch
    import vllm

    device = worker.model_runner.device
    return {
        "device": torch.cuda.get_device_name(device),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
    }


def _save_captures(worker, output: str) -> int:
    captures = worker.model_runner.model._exp586_captures
    values = np.concatenate(
        [
            capture.float().numpy() if hasattr(capture, "detach") else capture
            for capture in captures
        ],
        axis=0,
    )
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
    quantization: str | None = None,
    moe_backend: str = "auto",
    attention_backend: str | None = None,
    flash_attention_version: int | None = None,
    batch_size: int = 1,
    iterations: int = 8,
    validation_fixture: str | None = None,
    output_transfer_dtype: Literal["float32", "bfloat16"] = "float32",
    tokenizer_batch: bool = False,
    prefetch_tokenization: bool = False,
    minimal_sampling_head: bool = False,
    specialized_topk: bool = False,
    direct_sconv: bool = False,
    moe_config_json: str | None = None,
    chunked_prefill: bool = False,
    max_num_batched_tokens: int | None = None,
    in_process: bool = False,
    omp_threads: int | None = None,
    grug_cudagraph_mode: Literal["PIECEWISE", "FULL", "FULL_AND_PIECEWISE"]
    | None = None,
    async_output_transfer: bool = False,
    separate_output_transfer_stream: bool = False,
    deferred_output_transfer: bool = False,
    preallocated_output_transfer: bool = False,
    results_prefix: str | None = None,
) -> None:
    """Measure batched 8,192-token prefills and return dense A/C/G/T logits."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if (
        sum(
            (
                async_output_transfer,
                separate_output_transfer_stream,
                deferred_output_transfer,
                preallocated_output_transfer,
            )
        )
        > 1
    ):
        raise ValueError("Choose one asynchronous output-transfer implementation")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if grug_cudagraph_mode is not None:
        os.environ["EXP586_VLLM_CUDAGRAPH"] = "1"
        allowed_plugins = {
            name for name in os.environ.get("VLLM_PLUGINS", "").split(",") if name
        }
        allowed_plugins.add("exp586_grug_cudagraph")
        os.environ["VLLM_PLUGINS"] = ",".join(sorted(allowed_plugins))
    if in_process:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if omp_threads is not None:
        if omp_threads < 1:
            raise ValueError("omp_threads must be positive")
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    # vLLM reserves one generated token to trigger a standard engine prefill. The
    # model never evaluates position 8192; only the requested 8,192-token prompt.
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    from vllm import LLM, SamplingParams
    from vllm.config import CUDAGraphMode

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
    elif tokenizer_variant == "reduced_unigram":
        tokenizer = reduced_dna_unigram_tokenizer(tokenizer)
    elif tokenizer_variant != "full":
        raise ValueError(f"Unknown tokenizer variant: {tokenizer_variant}")
    base_ids = [tokenizer.token_to_id(base.lower()) for base in "ACGT"]
    if any(token_id is None for token_id in base_ids):
        raise ValueError("Tokenizer lacks isolated A/C/G/T tokens")
    resolved_base_ids = [int(token_id) for token_id in base_ids]
    moe_config = json.loads(moe_config_json) if moe_config_json is not None else None

    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        quantization=quantization,
        tensor_parallel_size=1,
        enforce_eager=not compile_engine,
        max_model_len=8193,
        max_num_batched_tokens=(
            max_num_batched_tokens
            if max_num_batched_tokens is not None
            else batch_size * 8193
        ),
        max_num_seqs=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        enable_prefix_caching=False,
        enable_chunked_prefill=chunked_prefill,
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
        compilation_config=(
            {"cudagraph_mode": CUDAGraphMode[grug_cudagraph_mode]}
            if grug_cudagraph_mode is not None
            else None
        ),
    )
    installed = llm.llm_engine.collective_rpc(
        _install_capture,
        args=(
            resolved_base_ids,
            variant,
            output_transfer_dtype,
            minimal_sampling_head,
            specialized_topk,
            direct_sconv,
            moe_config,
            async_output_transfer,
            separate_output_transfer_stream,
            deferred_output_transfer,
            preallocated_output_transfer,
            batch_size,
        ),
    )
    if installed != [True]:
        raise RuntimeError(f"Capture hook was not installed: {installed}")
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)

    def batch_at(index: int) -> list[str]:
        start = (index * batch_size) % len(sequences)
        return [
            sequences[(start + offset) % len(sequences)] for offset in range(batch_size)
        ]

    def tokenize(batch: list[str]) -> tuple[list[dict[str, list[int]]], float]:
        began = time.perf_counter()
        if tokenizer_batch:
            ids = encode_dna_batch_lists(tokenizer, batch)
            prompts = [{"prompt_token_ids": sequence_ids} for sequence_ids in ids]
        else:
            prompts = [
                {"prompt_token_ids": encode_dna(tokenizer, sequence).tolist()}
                for sequence in batch
            ]
        return prompts, time.perf_counter() - began

    def forward(
        prompts: list[dict[str, list[int]]], *, retain_captures: bool = False
    ) -> float:
        began = time.perf_counter()
        llm.generate(
            prompts,
            sampling,
            use_tqdm=False,
        )
        if (
            async_output_transfer
            or separate_output_transfer_stream
            or preallocated_output_transfer
        ):
            llm.llm_engine.collective_rpc(
                _finish_async_captures,
                args=(not retain_captures,),
            )
        elif deferred_output_transfer:
            llm.llm_engine.collective_rpc(
                _finish_deferred_captures,
                args=(not retain_captures,),
            )

        return time.perf_counter() - began

    def run(batch: list[str]) -> None:
        prompts, _ = tokenize(batch)
        forward(prompts)

    run(batch_at(0))
    llm.llm_engine.collective_rpc(_clear_captures)
    reset = llm.llm_engine.collective_rpc(_reset_peak_memory)
    if reset != [True]:
        raise RuntimeError(f"Peak-memory reset failed: {reset}")

    latencies: list[float] = []
    tokenization_latencies: list[float] = []
    forward_output_latencies: list[float] = []
    pipeline_elapsed: float | None = None
    if prefetch_tokenization:
        pipeline_began = time.perf_counter()
        completion_times: list[float] = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(tokenize, batch_at(0))
            for index in range(iterations):
                prompts, tokenization_latency = pending.result()
                if index + 1 < iterations:
                    pending = pool.submit(tokenize, batch_at(index + 1))
                forward_latency = forward(prompts)
                completion_times.append(time.perf_counter())
                tokenization_latencies.append(tokenization_latency)
                forward_output_latencies.append(forward_latency)
        pipeline_elapsed = time.perf_counter() - pipeline_began
        latencies = [
            completed - previous for previous, completed in pairwise(completion_times)
        ]
    else:
        for index in range(iterations):
            began = time.perf_counter()
            prompts, tokenization_latency = tokenize(batch_at(index))
            forward_latency = forward(prompts)
            tokenization_latencies.append(tokenization_latency)
            forward_output_latencies.append(forward_latency)
            latencies.append(time.perf_counter() - began)

    # Discard timed captures and validate every batch size on the same complete
    # fixture. This keeps throughput batch size separate from sample coverage.
    llm.llm_engine.collective_rpc(_clear_captures)
    validation_batch_size = min(batch_size, len(validation_sequences))
    for start in range(0, len(validation_sequences), validation_batch_size):
        prompts, _ = tokenize(
            validation_sequences[start : start + validation_batch_size]
        )
        forward(prompts, retain_captures=True)

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
    runtime = llm.llm_engine.collective_rpc(_runtime_info)
    if len(runtime) != 1:
        raise RuntimeError(f"Expected one vLLM worker, got {runtime}")

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
        "tokenization_latency_seconds": tokenization_latencies,
        "forward_output_latency_seconds": forward_output_latencies,
        "pipeline_elapsed_seconds": pipeline_elapsed,
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
        "quantization": quantization,
        "model_dir": model_dir,
        "model_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_variant": tokenizer_variant,
        "moe_backend": moe_backend,
        "attention_backend": attention_backend,
        "flash_attention_version": flash_attention_version,
        "output_transfer_dtype": output_transfer_dtype,
        "tokenizer_batch": tokenizer_batch,
        "prefetch_tokenization": prefetch_tokenization,
        "minimal_sampling_head": minimal_sampling_head,
        "specialized_topk": specialized_topk,
        "direct_sconv": direct_sconv,
        "moe_config": moe_config,
        "chunked_prefill": chunked_prefill,
        "max_num_batched_tokens": max_num_batched_tokens,
        "in_process": in_process,
        "omp_threads": omp_threads,
        "grug_cudagraph_mode": grug_cudagraph_mode,
        "async_output_transfer": async_output_transfer,
        "separate_output_transfer_stream": separate_output_transfer_stream,
        "deferred_output_transfer": deferred_output_transfer,
        "preallocated_output_transfer": preallocated_output_transfer,
        **runtime[0],
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
                f"quantization={quantization}; "
                f"moe={moe_backend}; attention={attention_backend or 'auto'}; "
                f"flash_attention_version={flash_attention_version or 'auto'}; "
                f"output_transfer_dtype={output_transfer_dtype}; batch={batch_size}"
                f"; tokenizer_batch={tokenizer_batch}; prefetch={prefetch_tokenization}"
                f"; minimal_sampling_head={minimal_sampling_head}"
                f"; specialized_topk={specialized_topk}; direct_sconv={direct_sconv}"
                f"; moe_config={moe_config}"
                f"; chunked_prefill={chunked_prefill}"
                f"; max_num_batched_tokens={max_num_batched_tokens}"
                f"; in_process={in_process}; omp_threads={omp_threads}"
                f"; grug_cudagraph_mode={grug_cudagraph_mode}"
                f"; async_output_transfer={async_output_transfer}"
                f"; separate_output_transfer_stream={separate_output_transfer_stream}"
                f"; deferred_output_transfer={deferred_output_transfer}"
                f"; preallocated_output_transfer={preallocated_output_transfer}"
            ),
            tokens_per_sequence=tokens_per_sequence,
            batch_size=batch_size,
        ),
    )
    if results_prefix is not None:
        upload_trial_artifacts(output_path, Path(ledger), results_prefix)
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
        "--tokenizer-variant",
        choices=("full", "reduced", "reduced_unigram"),
        default="full",
    )
    parser.add_argument("--compile-engine", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--quantization", choices=("fp8",))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--validation-fixture")
    parser.add_argument("--results-prefix")
    parser.add_argument(
        "--output-transfer-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
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
    parser.add_argument("--tokenizer-batch", action="store_true")
    parser.add_argument("--prefetch-tokenization", action="store_true")
    parser.add_argument("--minimal-sampling-head", action="store_true")
    parser.add_argument("--specialized-topk", action="store_true")
    parser.add_argument("--direct-sconv", action="store_true")
    parser.add_argument("--moe-config-json")
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--omp-threads", type=int)
    parser.add_argument(
        "--grug-cudagraph-mode",
        choices=("PIECEWISE", "FULL", "FULL_AND_PIECEWISE"),
    )
    parser.add_argument("--async-output-transfer", action="store_true")
    parser.add_argument("--separate-output-transfer-stream", action="store_true")
    parser.add_argument("--deferred-output-transfer", action="store_true")
    parser.add_argument("--preallocated-output-transfer", action="store_true")
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
        quantization=arguments.quantization,
        moe_backend=arguments.moe_backend,
        attention_backend=arguments.attention_backend,
        flash_attention_version=arguments.flash_attention_version,
        batch_size=arguments.batch_size,
        iterations=arguments.iterations,
        validation_fixture=arguments.validation_fixture,
        output_transfer_dtype=arguments.output_transfer_dtype,
        tokenizer_batch=arguments.tokenizer_batch,
        prefetch_tokenization=arguments.prefetch_tokenization,
        minimal_sampling_head=arguments.minimal_sampling_head,
        specialized_topk=arguments.specialized_topk,
        direct_sconv=arguments.direct_sconv,
        moe_config_json=arguments.moe_config_json,
        chunked_prefill=arguments.chunked_prefill,
        max_num_batched_tokens=arguments.max_num_batched_tokens,
        in_process=arguments.in_process,
        omp_threads=arguments.omp_threads,
        grug_cudagraph_mode=arguments.grug_cudagraph_mode,
        async_output_transfer=arguments.async_output_transfer,
        separate_output_transfer_stream=arguments.separate_output_transfer_stream,
        deferred_output_transfer=arguments.deferred_output_transfer,
        preallocated_output_transfer=arguments.preallocated_output_transfer,
        results_prefix=arguments.results_prefix,
    )
