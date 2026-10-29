"""Measure native JAX inference on one H100 with real PlantCAD windows."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path

import jax
import numpy as np
from levanter.grug.attention import GrugAttentionImplementation
from levanter.grug.grug_moe import MoeImplementation

from exp586_moe.data import pretrained_dna_tokenizer
from exp586_moe.inference.correctness import acgt_parity
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.jax_backend import (
    JaxOutputDtype,
    compile_forward,
    configure_fa4_forward,
    configure_sconv_tiles,
    encode,
    load_model,
)
from exp586_moe.inference.ledger import Trial, append_trial
from exp586_moe.inference.results_store import upload_trial_artifacts
from exp586_moe.inference.tokenizer import compact_dna_tokenizer, reduced_dna_tokenizer


def benchmark(
    fixture: str,
    ledger: str,
    output: str,
    *,
    variant: str = "full_head",
    name: str = "baseline",
    accepted: bool = True,
    iterations: int = 4,
    reference: str | None = None,
    tokenizer_variant: str = "full",
    batch_size: int = 1,
    validation_fixture: str | None = None,
    compact_vocab: bool = False,
    moe_implementation: MoeImplementation = "sonic",
    attention_implementation: GrugAttentionImplementation = "gpu_fa4_cute",
    expert_chunks: int = 1,
    output_dtype: JaxOutputDtype = "float32",
    prefetch_tokenization: bool = False,
    sconv_chunk: int | None = None,
    sconv_max_channel_block: int | None = None,
    sconv_num_warps: int | None = None,
    sconv_rows_per_step: int | None = None,
    sconv_exact_reference_rounding: bool = True,
    layer_scan_unroll: int = 1,
    remat_layers: bool = True,
    fa4_forward_tile_m: int | None = None,
    fa4_forward_tile_n: int | None = None,
    fa4_num_threads: int | None = None,
    profile_dir: str | None = None,
    results_prefix: str | None = None,
) -> None:
    """Measure complete batched scoring, including tokenization and host output."""
    if iterations < 2:
        raise ValueError("Use at least two timed iterations")
    began_productive = time.perf_counter()
    sequences = load_fixture(Path(fixture))
    validation_sequences = load_fixture(
        Path(validation_fixture) if validation_fixture is not None else Path(fixture)
    )
    if batch_size < 1 or batch_size > len(sequences):
        raise ValueError("Batch size must fit the unique-sequence fixture")
    tokenizer = compact_dna_tokenizer() if compact_vocab else pretrained_dna_tokenizer()
    if tokenizer_variant == "reduced" and not compact_vocab:
        tokenizer = reduced_dna_tokenizer(tokenizer)
    elif tokenizer_variant != "full":
        raise ValueError(f"Unknown tokenizer variant: {tokenizer_variant}")
    if compact_vocab and variant != "acgt_head":
        raise ValueError("Compact vocabulary requires the A/C/G/T output head")
    if layer_scan_unroll < 1:
        raise ValueError("layer_scan_unroll must be positive")
    configure_sconv_tiles(
        chunk=sconv_chunk,
        max_channel_block=sconv_max_channel_block,
        num_warps=sconv_num_warps,
        rows_per_step=sconv_rows_per_step,
        exact_reference_rounding=sconv_exact_reference_rounding,
    )
    configure_fa4_forward(
        tile_m=fa4_forward_tile_m,
        tile_n=fa4_forward_tile_n,
        num_threads=fa4_num_threads,
    )
    model, mesh, base_ids = load_model(
        compact_vocab=compact_vocab,
        moe_implementation=moe_implementation,
        attention_implementation=attention_implementation,
        expert_chunks=expert_chunks,
    )
    forward = compile_forward(
        model,
        mesh,
        base_ids,
        variant=variant,  # type: ignore[arg-type]
        output_dtype=output_dtype,
        layer_scan_unroll=layer_scan_unroll,
        remat_layers=remat_layers,
    )

    def batch_at(index: int) -> list[str]:
        start = (index * batch_size) % len(sequences)
        return [
            sequences[(start + offset) % len(sequences)] for offset in range(batch_size)
        ]

    warmup_ids = encode(tokenizer, batch_at(0))
    warmup = forward(jax.device_put(warmup_ids))
    jax.block_until_ready(warmup)
    del warmup

    latencies: list[float] = []
    tokenization_latencies: list[float] = []
    forward_output_latencies: list[float] = []
    pipeline_elapsed: float | None = None

    def timed_encode(batch: list[str]) -> tuple[np.ndarray, float]:
        began = time.perf_counter()
        ids = encode(tokenizer, batch)
        return ids, time.perf_counter() - began

    profile = (
        jax.profiler.trace(profile_dir) if profile_dir is not None else nullcontext()
    )
    with profile:
        if prefetch_tokenization:
            pipeline_began = time.perf_counter()
            completion_times: list[float] = []
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(timed_encode, batch_at(0))
                for index in range(iterations):
                    token_ids, tokenization_latency = pending.result()
                    if index + 1 < iterations:
                        pending = pool.submit(timed_encode, batch_at(index + 1))
                    began_forward = time.perf_counter()
                    logits = np.asarray(forward(jax.device_put(token_ids)))
                    result = logits[..., base_ids] if variant == "full_head" else logits
                    forward_output_latencies.append(time.perf_counter() - began_forward)
                    completion_times.append(time.perf_counter())
                    tokenization_latencies.append(tokenization_latency)
                    del logits, result
            pipeline_elapsed = time.perf_counter() - pipeline_began
            # Exclude the one-time pipeline fill. Adjacent output completions measure
            # sustainable throughput and remain comparable across iteration counts.
            latencies = [
                completed - previous
                for previous, completed in pairwise(completion_times)
            ]
        else:
            for index in range(iterations):
                batch = batch_at(index)
                began = time.perf_counter()
                token_ids = encode(tokenizer, batch)
                tokenized = time.perf_counter()
                logits = np.asarray(forward(jax.device_put(token_ids)))
                result = logits[..., base_ids] if variant == "full_head" else logits
                completed = time.perf_counter()
                tokenization_latencies.append(tokenized - began)
                forward_output_latencies.append(completed - tokenized)
                latencies.append(completed - began)
                del logits, result

    # Use the same complete fixture for every batch-size correctness decision.
    validation_parts = []
    validation_batch_size = min(batch_size, len(validation_sequences))
    for start in range(0, len(validation_sequences), validation_batch_size):
        token_ids = encode(
            tokenizer, validation_sequences[start : start + validation_batch_size]
        )
        logits = np.asarray(forward(jax.device_put(token_ids)))
        result = logits[..., base_ids] if variant == "full_head" else logits
        validation_parts.append(np.asarray(result, dtype=np.float32))
    representative = np.concatenate(validation_parts, axis=0)

    parity: dict[str, float | bool | int] = {
        "accepted": True,
        "raw_max_abs_error": 0.0,
    }
    if reference is not None:
        expected = np.load(reference)["logits"]
        if expected.shape[0] != len(validation_sequences):
            raise ValueError("Correctness reference must cover the complete fixture")
        parity = acgt_parity(
            representative,
            expected,
            validation_sequences,
        )
    else:
        reference_path = Path(output).with_suffix(".reference.npz")
        np.savez_compressed(reference_path, logits=representative)

    elapsed = time.perf_counter() - began_productive
    median = float(np.median(latencies))
    sequences_per_second = batch_size / median
    tokens_per_sequence = 8192
    memory_stats = jax.devices()[0].memory_stats() or {}
    result_payload = {
        "backend": "jax",
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
        "accepted": bool(accepted) and bool(parity["accepted"]),
        "jax_version": jax.__version__,
        "device": jax.devices()[0].device_kind,
        "tokenizer_variant": tokenizer_variant,
        "compact_vocab": compact_vocab,
        "moe_implementation": moe_implementation,
        "attention_implementation": attention_implementation,
        "expert_chunks": expert_chunks,
        "output_dtype": output_dtype,
        "prefetch_tokenization": prefetch_tokenization,
        "sconv_chunk": sconv_chunk,
        "sconv_max_channel_block": sconv_max_channel_block,
        "sconv_num_warps": sconv_num_warps,
        "sconv_rows_per_step": sconv_rows_per_step,
        "sconv_exact_reference_rounding": sconv_exact_reference_rounding,
        "layer_scan_unroll": layer_scan_unroll,
        "remat_layers": remat_layers,
        "fa4_forward_tile_m": fa4_forward_tile_m,
        "fa4_forward_tile_n": fa4_forward_tile_n,
        "fa4_num_threads": fa4_num_threads,
        "profile_dir": profile_dir,
        "peak_memory_gib": memory_stats.get("peak_bytes_in_use", 0) / 2**30,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_payload, indent=2) + "\n")
    append_trial(
        Path(ledger),
        Trial.now(
            backend="jax",
            name=name,
            productive_seconds=elapsed,
            sequences_per_second=sequences_per_second,
            median_latency_seconds=median,
            accepted=bool(result_payload["accepted"]),
            correctness_max_abs_error=float(parity["raw_max_abs_error"]),
            notes=(
                f"{variant}; tokenizer={tokenizer_variant}; "
                f"compact_vocab={compact_vocab}; moe={moe_implementation}; "
                f"attention={attention_implementation}; expert_chunks={expert_chunks}; "
                f"output_dtype={output_dtype}; prefetch={prefetch_tokenization}; "
                f"sconv=({sconv_chunk},{sconv_max_channel_block},"
                f"{sconv_num_warps},{sconv_rows_per_step},"
                f"exact={sconv_exact_reference_rounding}); "
                f"scan_unroll={layer_scan_unroll}; remat={remat_layers}; "
                f"fa4=({fa4_forward_tile_m},{fa4_forward_tile_n},"
                f"threads={fa4_num_threads}); "
                f"batch={batch_size}"
            ),
            tokens_per_sequence=tokens_per_sequence,
            batch_size=batch_size,
        ),
    )
    if results_prefix is not None:
        upload_trial_artifacts(output_path, Path(ledger), results_prefix)
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
