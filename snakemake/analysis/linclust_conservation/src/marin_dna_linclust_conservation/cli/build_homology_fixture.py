"""Build a small projected-ortholog fixture from immutable S3 objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3

from marin_dna_linclust_conservation.homology_fixture import (
    download_and_build_center_expanded_projection_fixture,
    download_and_build_projection_fixture,
    parse_sources_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--max-anchors", type=int, required=True)
    parser.add_argument("--candidate-anchors", type=int, required=True)
    parser.add_argument(
        "--sequence-mode",
        choices=("embedded", "center-expanded"),
        default="embedded",
    )
    parser.add_argument("--selection-window-length", type=int)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    sources = parse_sources_json(args.sources)
    s3_client = boto3.client("s3")
    if args.sequence_mode == "embedded":
        assert args.selection_window_length is None
        receipt = download_and_build_projection_fixture(
            sources=sources,
            s3_client=s3_client,
            max_anchors=args.max_anchors,
            candidate_anchors=args.candidate_anchors,
            window_length=args.window_length,
            fasta_path=args.fasta,
            truth_path=args.truth,
        )
    else:
        assert args.sequence_mode == "center-expanded"
        assert args.selection_window_length is not None
        receipt = download_and_build_center_expanded_projection_fixture(
            sources=sources,
            s3_client=s3_client,
            max_anchors=args.max_anchors,
            candidate_anchors=args.candidate_anchors,
            selection_window_length=args.selection_window_length,
            window_length=args.window_length,
            fasta_path=args.fasta,
            truth_path=args.truth,
        )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
