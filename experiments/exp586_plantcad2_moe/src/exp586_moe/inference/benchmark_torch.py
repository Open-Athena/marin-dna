"""Measure plain eager PyTorch inference on one H100."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from exp586_moe.inference.contract import token_contract
from exp586_moe.inference.correctness import acgt_parity
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.ledger import Trial, append_trial
from exp586_moe.inference.tokenizer import encode_dna, reduced_dna_tokenizer
from exp586_moe.inference.torch_backend import EagerGrugMoE, TorchVariant


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
    tokenizer_variant: str = "wrapper",
) -> None:
    """Run batch-one end-to-end trials, including HF tokenization and host output."""
    if iterations < 2:
        raise ValueError("Use at least two timed iterations")
    began_productive = time.perf_counter()
    device = torch.device("cuda:0")
    directory = Path(model_dir)
    sequences = load_fixture(Path(fixture))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
    if tokenizer_variant not in {"wrapper", "backend", "reduced"}:
        raise ValueError(f"Unknown tokenizer variant: {tokenizer_variant}")
    backend_tokenizer = tokenizer.backend_tokenizer
    if tokenizer_variant == "reduced":
        backend_tokenizer = reduced_dna_tokenizer(backend_tokenizer)
    contract = token_contract(tokenizer.backend_tokenizer)
    base_ids = [contract.output_ids[base] for base in "ACGT"]
    model = EagerGrugMoE(directory, device)

    def forward(token_ids: torch.Tensor) -> torch.Tensor:
        return model(token_ids, base_ids, variant=variant)

    model_forward = (
        torch.compile(forward, fullgraph=True, dynamic=False, mode="reduce-overhead")
        if compile_model
        else forward
    )

    def run(sequence: str) -> np.ndarray:
        if tokenizer_variant != "wrapper":
            ids = encode_dna(backend_tokenizer, sequence)
            token_ids = torch.from_numpy(ids).to(device).unsqueeze(0)
        else:
            encoded = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
            token_ids = encoded["input_ids"].to(device)
        with torch.inference_mode():
            result = model_forward(token_ids)
        return result.float().cpu().numpy()

    run(sequences[0])
    torch.cuda.synchronize()
    latencies: list[float] = []
    representative: np.ndarray | None = None
    for index in range(iterations):
        began = time.perf_counter()
        result = run(sequences[index % len(sequences)])
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - began)
        if representative is None:
            representative = result
    assert representative is not None

    parity: dict[str, float | bool | int] = {
        "accepted": True,
        "raw_max_abs_error": 0.0,
    }
    if reference is not None:
        expected = np.load(reference)["logits"]
        if expected.shape[0] == 1:
            actual = representative
            validation_sequences = sequences[:1]
        elif expected.shape[0] == len(sequences):
            actual = np.concatenate([run(sequence) for sequence in sequences])
            validation_sequences = sequences
        else:
            raise ValueError("Correctness reference batch differs")
        parity = acgt_parity(actual, expected, validation_sequences)
        np.savez_compressed(Path(output).with_suffix(".reference.npz"), logits=actual)
    effective_accepted = bool(accepted) and bool(parity["accepted"])
    elapsed = time.perf_counter() - began_productive
    median = float(np.median(latencies))
    sequences_per_second = 1 / median
    tokens_per_sequence = 8192
    payload = {
        "backend": "torch",
        "variant": variant,
        "name": name,
        "batch_size": 1,
        "sequence_length": 8192,
        "iterations": iterations,
        "latency_seconds": latencies,
        "median_latency_seconds": median,
        "tokens_per_sequence": tokens_per_sequence,
        "input_tokens_per_second": sequences_per_second * tokens_per_sequence,
        "sequences_per_second": sequences_per_second,
        "productive_seconds": elapsed,
        "parity": parity,
        "accepted": effective_accepted,
        "compiled": compile_model,
        "tokenizer_variant": tokenizer_variant,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
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
            notes=variant,
            tokens_per_sequence=tokens_per_sequence,
        ),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
