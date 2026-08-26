"""Build a truth-plus-real-background scaling FASTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3

from marin_dna_linclust_conservation.background_scaling import (
    BackgroundFastaSource,
    build_background_fixture,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=json.loads, required=True)
    parser.add_argument("--records-per-source", type=json.loads, required=True)
    parser.add_argument("--truth-fasta", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    receipt = build_background_fixture(
        sources=[BackgroundFastaSource.from_dict(row) for row in args.sources],
        records_per_source=args.records_per_source,
        truth_fasta_path=args.truth_fasta,
        output_fasta_path=args.output_fasta,
        s3_client=boto3.client("s3"),
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
