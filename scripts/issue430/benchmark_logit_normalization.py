"""Microbenchmark ACGT-only versus full-vocabulary logit normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F


BYTES_PER_GIB = 1024**3


def _nucleotide_log_softmax(
    logits: torch.Tensor, nucleotide_token_ids: torch.Tensor
) -> torch.Tensor:
    return F.log_softmax(logits[..., nucleotide_token_ids].float(), dim=-1)


def _full_vocab_log_softmax(
    logits: torch.Tensor, nucleotide_token_ids: torch.Tensor
) -> torch.Tensor:
    del nucleotide_token_ids
    return F.log_softmax(logits.float(), dim=-1)


def _measure_milliseconds(
    function: object,
    logits: torch.Tensor,
    nucleotide_token_ids: torch.Tensor,
    *,
    iterations: int,
    rounds: int,
) -> list[float]:
    timings: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = function(logits, nucleotide_token_ids)
        end.record()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        timings.append(start.elapsed_time(end) / iterations)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=896)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    assert args.batch_size > 0
    assert args.sequence_length > 0
    assert args.vocab_size == 7, "issue #430 checkpoint has a seven-token vocabulary"
    assert args.iterations > 0
    assert args.rounds >= 3

    torch.manual_seed(430)
    logits = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.vocab_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    nucleotide_token_ids = torch.tensor([3, 4, 5, 6], device="cuda")
    nucleotide = torch.compile(_nucleotide_log_softmax, mode="default", fullgraph=True)
    full_vocab = torch.compile(_full_vocab_log_softmax, mode="default", fullgraph=True)
    nucleotide(logits, nucleotide_token_ids)
    full_vocab(logits, nucleotide_token_ids)
    torch.cuda.synchronize()

    nucleotide_ms = _measure_milliseconds(
        nucleotide,
        logits,
        nucleotide_token_ids,
        iterations=args.iterations,
        rounds=args.rounds,
    )
    full_vocab_ms = _measure_milliseconds(
        full_vocab,
        logits,
        nucleotide_token_ids,
        iterations=args.iterations,
        rounds=args.rounds,
    )
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shape": list(logits.shape),
        "dtype": str(logits.dtype),
        "iterations_per_round": args.iterations,
        "nucleotide_milliseconds": nucleotide_ms,
        "full_vocab_milliseconds": full_vocab_ms,
        "nucleotide_median_milliseconds": median(nucleotide_ms),
        "full_vocab_median_milliseconds": median(full_vocab_ms),
        "full_vocab_over_nucleotide": median(full_vocab_ms) / median(nucleotide_ms),
        "input_gib": logits.numel() * logits.element_size() / BYTES_PER_GIB,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
