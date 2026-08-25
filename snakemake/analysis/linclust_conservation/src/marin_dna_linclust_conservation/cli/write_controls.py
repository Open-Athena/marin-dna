"""Write a deterministic ordering of the MMseqs2 synthetic controls."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from marin_dna_linclust_conservation.controls import synthetic_sequences, write_fasta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    args = parser.parse_args()

    records = synthetic_sequences()
    ordered = dict(
        sorted(
            records.items(),
            key=lambda item: (
                hashlib.sha256(f"{args.order_seed}:{item[0]}".encode()).digest(),
                item[0],
            ),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(ordered, str(args.output))


if __name__ == "__main__":
    main()
