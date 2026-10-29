"""Measure native JAX inference on one H100 with real PlantCAD windows."""

import json
import time
from pathlib import Path

import jax
import numpy as np

from exp586_moe.data import pretrained_dna_tokenizer
from exp586_moe.inference.correctness import acgt_parity
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.jax_backend import compile_forward, encode, load_model
from exp586_moe.inference.ledger import Trial, append_trial
from exp586_moe.inference.tokenizer import reduced_dna_tokenizer


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
    tokenizer = pretrained_dna_tokenizer()
    if tokenizer_variant == "reduced":
        tokenizer = reduced_dna_tokenizer(tokenizer)
    elif tokenizer_variant != "full":
        raise ValueError(f"Unknown tokenizer variant: {tokenizer_variant}")
    model, mesh, base_ids = load_model()
    forward = compile_forward(model, mesh, base_ids, variant=variant)  # type: ignore[arg-type]

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
    for index in range(iterations):
        batch = batch_at(index)
        began = time.perf_counter()
        token_ids = encode(tokenizer, batch)
        logits = np.asarray(forward(jax.device_put(token_ids)))
        result = logits[..., base_ids] if variant == "full_head" else logits
        latencies.append(time.perf_counter() - began)
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
    result_payload = {
        "backend": "jax",
        "variant": variant,
        "name": name,
        "batch_size": batch_size,
        "sequence_length": 8192,
        "iterations": iterations,
        "latency_seconds": latencies,
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
            notes=f"{variant}; tokenizer={tokenizer_variant}; batch={batch_size}",
            tokens_per_sequence=tokens_per_sequence,
            batch_size=batch_size,
        ),
    )
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
