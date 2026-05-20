#!/usr/bin/env python
"""Create a DNA character-level tokenizer and upload it to HuggingFace Hub."""

import argparse

from marin_dna.tokenizer.char import create_char_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a DNA character-level tokenizer and upload it to HuggingFace Hub."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo id to upload to (e.g. bolinas-dna/tokenizer-char).",
    )
    parser.add_argument(
        "--bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include [BOS] token (default: True). Use --no-bos to disable.",
    )
    parser.add_argument(
        "--eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include [EOS] token (default: True). Use --no-eos to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tok = create_char_tokenizer(bos=args.bos, eos=args.eos)
    tok.push_to_hub(args.repo)


if __name__ == "__main__":
    main()
