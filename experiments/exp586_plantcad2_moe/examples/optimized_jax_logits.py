"""Generate dense A/C/G/T logits with the optimized native-JAX path."""

import argparse
from pathlib import Path

import numpy as np

from exp586_moe.inference.optimized_jax import OptimizedJaxDnaLogits


def main(sequences_file: Path, output_dir: Path, batch_size: int) -> None:
    """Read one 8,192-base ACGTN sequence per line and save batched logits."""
    sequences = [line.strip() for line in sequences_file.read_text().splitlines()]
    if not sequences:
        raise ValueError("The sequence file is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = OptimizedJaxDnaLogits(batch_size=batch_size)
    scorer.warmup(sequences[0])
    for index, logits in enumerate(scorer.score_batches(sequences)):
        output = output_dir / f"logits-{index:05d}.npy"
        np.save(output, logits)
        print(f"{output}: {logits.shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=352)
    arguments = parser.parse_args()
    main(arguments.sequences_file, arguments.output_dir, arguments.batch_size)
