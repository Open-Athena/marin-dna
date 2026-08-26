"""Build one bounded species-aware seed graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna_linclust_conservation.seed_graph import (
    SeedGraphConfiguration,
    build_seed_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--version", type=Path, required=True)
    parser.add_argument("--kmer-length", type=int, required=True)
    parser.add_argument("--selected-seeds", type=int, required=True)
    parser.add_argument("--max-seed-frequency", type=int, required=True)
    parser.add_argument("--min-shared-seeds", type=int, required=True)
    parser.add_argument("--hash-seed", type=int, required=True)
    parser.add_argument("--source-aliases", type=json.loads, default={})
    args = parser.parse_args()

    configuration = SeedGraphConfiguration(
        kmer_length=args.kmer_length,
        selected_seeds_per_sequence=args.selected_seeds,
        max_seed_frequency=args.max_seed_frequency,
        min_shared_seeds=args.min_shared_seeds,
        hash_seed=args.hash_seed,
    )
    receipt = build_seed_graph(
        fasta_path=args.fasta,
        truth_path=args.truth,
        assignments_path=args.assignments,
        configuration=configuration,
        source_aliases=args.source_aliases,
    )
    receipt["configuration"] = {
        "hash_seed": configuration.hash_seed,
        "kmer_length": configuration.kmer_length,
        "max_seed_frequency": configuration.max_seed_frequency,
        "min_shared_seeds": configuration.min_shared_seeds,
        "selected_seeds_per_sequence": configuration.selected_seeds_per_sequence,
    }
    receipt["source_aliases"] = dict(sorted(args.source_aliases.items()))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    args.version.write_text("marin species-aware repeat-capped seed graph v1\n")


if __name__ == "__main__":
    main()
