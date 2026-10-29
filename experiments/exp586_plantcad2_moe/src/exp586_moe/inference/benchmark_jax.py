"""Measure native JAX inference on one H100 with real PlantCAD windows."""

import json
import time
from pathlib import Path

import jax
import numpy as np

from exp586_moe.data import pretrained_dna_tokenizer
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.jax_backend import compile_forward, encode, load_model
from exp586_moe.inference.ledger import Trial, append_trial


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
) -> None:
    """Run batch-one end-to-end trials, including HF tokenization and host output."""
    if iterations < 2:
        raise ValueError("Use at least two timed iterations")
    began_productive = time.perf_counter()
    sequences = load_fixture(Path(fixture))
    tokenizer = pretrained_dna_tokenizer()
    model, mesh, base_ids = load_model()
    forward = compile_forward(model, mesh, base_ids, variant=variant)  # type: ignore[arg-type]

    warmup_ids = encode(tokenizer, [sequences[0]])
    warmup = forward(jax.device_put(warmup_ids))
    jax.block_until_ready(warmup)
    del warmup

    latencies: list[float] = []
    representative: np.ndarray | None = None
    for index in range(iterations):
        sequence = sequences[index % len(sequences)]
        began = time.perf_counter()
        token_ids = encode(tokenizer, [sequence])
        logits = np.asarray(forward(jax.device_put(token_ids)))
        result = logits[..., base_ids] if variant == "full_head" else logits
        latencies.append(time.perf_counter() - began)
        if representative is None:
            representative = np.asarray(result, dtype=np.float32)
        del logits, result
    assert representative is not None

    error = 0.0
    if reference is not None:
        expected = np.load(reference)["logits"]
        if expected.shape != representative.shape:
            raise ValueError("Correctness reference shape differs")
        error = float(np.max(np.abs(expected - representative)))
    else:
        reference_path = Path(output).with_suffix(".reference.npz")
        np.savez_compressed(reference_path, logits=representative)

    elapsed = time.perf_counter() - began_productive
    median = float(np.median(latencies))
    result_payload = {
        "backend": "jax",
        "variant": variant,
        "name": name,
        "batch_size": 1,
        "sequence_length": 8192,
        "iterations": iterations,
        "latency_seconds": latencies,
        "median_latency_seconds": median,
        "sequences_per_second": 1 / median,
        "productive_seconds": elapsed,
        "correctness_max_abs_error": error,
        "jax_version": jax.__version__,
        "device": jax.devices()[0].device_kind,
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
            sequences_per_second=1 / median,
            median_latency_seconds=median,
            accepted=accepted,
            correctness_max_abs_error=error,
            notes=variant,
        ),
    )
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
