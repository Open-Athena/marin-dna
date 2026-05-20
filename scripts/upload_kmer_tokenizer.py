#!/usr/bin/env python
"""Create a DNA k-mer tokenizer and upload it to HuggingFace Hub."""

import argparse

from marin_dna.tokenizer.kmer import create_kmer_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a DNA k-mer tokenizer and upload it to HuggingFace Hub."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="K-mer size (e.g. 8).",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo id to upload to (e.g. bolinas-dna/tokenizer-8-mer).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tok = create_kmer_tokenizer(args.k)
    tok.push_to_hub(args.repo)


if __name__ == "__main__":
    main()
