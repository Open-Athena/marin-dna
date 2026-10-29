"""Measure plain eager PyTorch inference on one H100."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from exp586_moe.inference.contract import token_contract
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.ledger import Trial, append_trial
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
    correctness_atol: float = 0.25,
) -> None:
    """Run batch-one end-to-end trials, including HF tokenization and host output."""
    if iterations < 2:
        raise ValueError("Use at least two timed iterations")
    began_productive = time.perf_counter()
    device = torch.device("cuda:0")
    directory = Path(model_dir)
    sequences = load_fixture(Path(fixture))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
    contract = token_contract(tokenizer.backend_tokenizer)
    base_ids = [contract.output_ids[base] for base in "ACGT"]
    model = EagerGrugMoE(directory, device)

    def run(sequence: str) -> np.ndarray:
        encoded = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
        token_ids = encoded["input_ids"].to(device)
        result = model(token_ids, base_ids, variant=variant)
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

    error = 0.0
    if reference is not None:
        expected = np.load(reference)["logits"]
        if expected.shape != representative.shape:
            raise ValueError("Correctness reference shape differs")
        error = float(np.max(np.abs(expected - representative)))
    effective_accepted = accepted and (reference is None or error <= correctness_atol)
    elapsed = time.perf_counter() - began_productive
    median = float(np.median(latencies))
    payload = {
        "backend": "torch",
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
        "correctness_atol": correctness_atol,
        "accepted": effective_accepted,
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
            sequences_per_second=1 / median,
            median_latency_seconds=median,
            accepted=effective_accepted,
            correctness_max_abs_error=error,
            notes=variant,
        ),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
