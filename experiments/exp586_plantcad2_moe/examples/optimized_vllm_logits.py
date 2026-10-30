"""Generate dense A/C/G/T logits with the optimized vLLM path."""

import argparse
from pathlib import Path

from exp586_moe.inference.optimized_vllm import OptimizedVllmDnaLogits


def main(
    model_dir: Path, sequences_file: Path, output_dir: Path, batch_size: int
) -> None:
    """Read one 8,192-base ACGTN sequence per line and save batched logits."""
    sequences = [line.strip() for line in sequences_file.read_text().splitlines()]
    if not sequences:
        raise ValueError("The sequence file is empty")
    scorer = OptimizedVllmDnaLogits(model_dir, batch_size=batch_size)
    scorer.warmup(sequences[0])
    for output, shape in scorer.score_to_directory(sequences, output_dir):
        print(f"{output}: {shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sequences-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    arguments = parser.parse_args()
    main(
        arguments.model_dir,
        arguments.sequences_file,
        arguments.output_dir,
        arguments.batch_size,
    )
