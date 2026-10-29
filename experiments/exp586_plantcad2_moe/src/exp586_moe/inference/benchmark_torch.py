"""Measure plain eager PyTorch inference on one H100."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from exp586_moe.inference.contract import token_contract
from exp586_moe.inference.correctness import acgt_parity
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.ledger import Trial, append_trial
from exp586_moe.inference.results_store import upload_trial_artifacts
from exp586_moe.inference.tokenizer import (
    encode_dna,
    encode_dna_batch,
    reduced_dna_tokenizer,
    reduced_dna_unigram_tokenizer,
)
from exp586_moe.inference.torch_backend import (
    EagerGrugMoE,
    TorchAttentionBackend,
    TorchGlobalAttentionBackend,
    TorchSconvBackend,
    TorchVariant,
)


def benchmark(
    model_dir: str,
    fixture: str,
    ledger: str,
    output: str,
    *,
    variant: TorchVariant = "full_head",
    name: str = "baseline",
    accepted: bool = True,
    iterations: int = 2,
    reference: str | None = None,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    tokenizer_variant: str = "wrapper",
    attention_backend: TorchAttentionBackend = "sdpa_expanded",
    global_attention_backend: TorchGlobalAttentionBackend = "auto",
    global_attention_gqa: bool = False,
    batch_size: int = 1,
    validation_fixture: str | None = None,
    output_transfer_dtype: Literal["float32", "bfloat16"] = "float32",
    prefetch_tokenization: bool = False,
    moe_config_json: str | dict[str, int] | None = None,
    specialized_topk: bool = False,
    sconv_backend: TorchSconvBackend = "conv1d",
    tokenizer_batch: bool = False,
    profile_path: str | None = None,
    fp8_moe: bool = False,
    fp8_moe_ptpc: bool = False,
    results_prefix: str | None = None,
) -> None:
    """Measure complete batched scoring, including tokenization and host output."""
    if iterations < 2:
        raise ValueError("Use at least two timed iterations")
    began_productive = time.perf_counter()
    device = torch.device("cuda:0")
    directory = Path(model_dir)
    sequences = load_fixture(Path(fixture))
    validation_sequences = load_fixture(
        Path(validation_fixture) if validation_fixture is not None else Path(fixture)
    )
    if batch_size < 1 or batch_size > len(sequences):
        raise ValueError("Batch size must fit the unique-sequence fixture")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
    if tokenizer_variant not in {"wrapper", "backend", "reduced", "reduced_unigram"}:
        raise ValueError(f"Unknown tokenizer variant: {tokenizer_variant}")
    backend_tokenizer = tokenizer.backend_tokenizer
    if tokenizer_variant == "reduced":
        backend_tokenizer = reduced_dna_tokenizer(backend_tokenizer)
    elif tokenizer_variant == "reduced_unigram":
        backend_tokenizer = reduced_dna_unigram_tokenizer(backend_tokenizer)
    contract = token_contract(tokenizer.backend_tokenizer)
    base_ids = [contract.output_ids[base] for base in "ACGT"]
    model = EagerGrugMoE(
        directory,
        device,
        attention_backend=attention_backend,
        global_attention_backend=global_attention_backend,
        global_attention_gqa=global_attention_gqa,
        specialized_topk=specialized_topk,
        sconv_backend=sconv_backend,
        fp8_moe=fp8_moe,
        fp8_moe_ptpc=fp8_moe_ptpc,
    )

    def forward(token_ids: torch.Tensor) -> torch.Tensor:
        return model(token_ids, base_ids, variant=variant)

    compile_modes = torch._inductor.list_mode_options()
    if compile_model and compile_mode not in compile_modes:
        raise ValueError(
            f"Unknown torch.compile mode {compile_mode!r}; choose from {compile_modes}"
        )
    model_forward = (
        torch.compile(forward, fullgraph=True, dynamic=False, mode=compile_mode)
        if compile_model
        else forward
    )
    moe_config = (
        json.loads(moe_config_json)
        if isinstance(moe_config_json, str)
        else moe_config_json
    )
    if moe_config is not None:
        from vllm.model_executor.layers.fused_moe import (
            override_config as vllm_override_config,
        )
    else:
        vllm_override_config = None

    def tokenize(batch: list[str]) -> tuple[torch.Tensor, float]:
        began = time.perf_counter()
        if tokenizer_variant != "wrapper":
            ids = (
                encode_dna_batch(backend_tokenizer, batch)
                if tokenizer_batch
                else np.stack(
                    [encode_dna(backend_tokenizer, sequence) for sequence in batch]
                )
            )
            token_ids = torch.from_numpy(ids)
        else:
            encoded = tokenizer(batch, add_special_tokens=False, return_tensors="pt")
            token_ids = encoded["input_ids"]
        return token_ids, time.perf_counter() - began

    def forward_output(token_ids: torch.Tensor) -> tuple[np.ndarray, float]:
        began = time.perf_counter()
        token_ids = token_ids.to(device)
        config_context = (
            vllm_override_config(moe_config)
            if vllm_override_config is not None
            else nullcontext()
        )
        with torch.inference_mode(), config_context:
            result = model_forward(token_ids)
        if output_transfer_dtype == "float32":
            output = result.float().cpu().numpy()
        else:
            output = result.to(torch.bfloat16).cpu().float().numpy()
        return output, time.perf_counter() - began

    def run(batch: list[str]) -> np.ndarray:
        token_ids, _ = tokenize(batch)
        output, _ = forward_output(token_ids)
        return output

    def batch_at(index: int) -> list[str]:
        start = (index * batch_size) % len(sequences)
        return [
            sequences[(start + offset) % len(sequences)] for offset in range(batch_size)
        ]

    run(batch_at(0))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    latencies: list[float] = []
    tokenization_latencies: list[float] = []
    forward_output_latencies: list[float] = []
    pipeline_elapsed: float | None = None
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        if profile_path is not None
        else nullcontext()
    )
    with profiler as active_profiler:
        if prefetch_tokenization:
            pipeline_began = time.perf_counter()
            completion_times: list[float] = []
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(tokenize, batch_at(0))
                for index in range(iterations):
                    token_ids, tokenization_latency = pending.result()
                    if index + 1 < iterations:
                        pending = pool.submit(tokenize, batch_at(index + 1))
                    _, forward_output_latency = forward_output(token_ids)
                    torch.cuda.synchronize()
                    completion_times.append(time.perf_counter())
                    tokenization_latencies.append(tokenization_latency)
                    forward_output_latencies.append(forward_output_latency)
                    if profile_path is not None:
                        active_profiler.step()
            pipeline_elapsed = time.perf_counter() - pipeline_began
            latencies = [
                completed - previous
                for previous, completed in pairwise(completion_times)
            ]
        else:
            for index in range(iterations):
                began = time.perf_counter()
                token_ids, tokenization_latency = tokenize(batch_at(index))
                _, forward_output_latency = forward_output(token_ids)
                torch.cuda.synchronize()
                tokenization_latencies.append(tokenization_latency)
                forward_output_latencies.append(forward_output_latency)
                latencies.append(time.perf_counter() - began)
                if profile_path is not None:
                    active_profiler.step()
    if profile_path is not None:
        active_profiler.export_chrome_trace(profile_path)

    # Correctness always uses the complete fixture so acceptance cannot change
    # merely because a throughput trial used a larger validation sample.
    validation_parts = [
        run(
            validation_sequences[
                start : start + min(batch_size, len(validation_sequences))
            ]
        )
        for start in range(
            0, len(validation_sequences), min(batch_size, len(validation_sequences))
        )
    ]
    representative = np.concatenate(validation_parts, axis=0)

    parity: dict[str, float | bool | int] = {
        "accepted": True,
        "raw_max_abs_error": 0.0,
    }
    if reference is not None:
        expected = np.load(reference)["logits"]
        if expected.shape[0] != len(validation_sequences):
            raise ValueError("Correctness reference must cover the complete fixture")
        actual = representative
        parity = acgt_parity(actual, expected, validation_sequences)
        np.savez_compressed(Path(output).with_suffix(".reference.npz"), logits=actual)
    effective_accepted = bool(accepted) and bool(parity["accepted"])
    elapsed = time.perf_counter() - began_productive
    median = float(np.median(latencies))
    sequences_per_second = batch_size / median
    tokens_per_sequence = 8192
    payload = {
        "backend": "torch",
        "variant": variant,
        "name": name,
        "batch_size": batch_size,
        "sequence_length": 8192,
        "iterations": iterations,
        "latency_seconds": latencies,
        "tokenization_latency_seconds": tokenization_latencies,
        "forward_output_latency_seconds": forward_output_latencies,
        "pipeline_elapsed_seconds": pipeline_elapsed,
        "pipeline_including_fill_tokens_per_second": (
            batch_size * iterations * 8192 / pipeline_elapsed
            if pipeline_elapsed is not None
            else None
        ),
        "median_latency_seconds": median,
        "tokens_per_sequence": tokens_per_sequence,
        "input_tokens_per_second": sequences_per_second * tokens_per_sequence,
        "sequences_per_second": sequences_per_second,
        "productive_seconds": elapsed,
        "parity": parity,
        "accepted": effective_accepted,
        "compiled": compile_model,
        "compile_mode": compile_mode if compile_model else None,
        "tokenizer_variant": tokenizer_variant,
        "attention_backend": attention_backend,
        "global_attention_backend": global_attention_backend,
        "global_attention_gqa": global_attention_gqa,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "output_transfer_dtype": output_transfer_dtype,
        "prefetch_tokenization": prefetch_tokenization,
        "moe_config": moe_config,
        "specialized_topk": specialized_topk,
        "sconv_backend": sconv_backend,
        "tokenizer_batch": tokenizer_batch,
        "profile_path": profile_path,
        "fp8_moe": fp8_moe,
        "fp8_moe_ptpc": fp8_moe_ptpc,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    append_trial(
        Path(ledger),
        Trial.now(
            backend="torch",
            name=name,
            productive_seconds=elapsed,
            sequences_per_second=sequences_per_second,
            median_latency_seconds=median,
            accepted=effective_accepted,
            correctness_max_abs_error=float(parity["raw_max_abs_error"]),
            notes=(
                f"{variant}; compiled={compile_model}; "
                f"compile_mode={compile_mode if compile_model else None}; "
                f"tokenizer={tokenizer_variant}; attention={attention_backend}; "
                f"global_attention={global_attention_backend}; "
                f"global_gqa={global_attention_gqa}; "
                f"output_transfer_dtype={output_transfer_dtype}; "
                f"prefetch={prefetch_tokenization}; batch={batch_size}"
                f"; moe_config={moe_config}"
                f"; specialized_topk={specialized_topk}"
                f"; sconv_backend={sconv_backend}"
                f"; tokenizer_batch={tokenizer_batch}"
                f"; fp8_moe={fp8_moe}; fp8_moe_ptpc={fp8_moe_ptpc}"
            ),
            tokens_per_sequence=tokens_per_sequence,
            batch_size=batch_size,
        ),
    )
    if results_prefix is not None:
        upload_trial_artifacts(output_path, Path(ledger), results_prefix)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
