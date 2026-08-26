"""Prepare the bounded FASTA and pair labels for a seed-graph alignment diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna_linclust_conservation.seed_alignment import (
    prepare_seed_alignment_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--subset-fasta", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_seed_alignment_subset(
        fasta_path=args.fasta,
        truth_path=args.truth,
        assignments_path=args.assignments,
        subset_fasta_path=args.subset_fasta,
        pairs_path=args.pairs,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
