"""Conda-activated command boundary for batched 2bit sequence extraction."""

from __future__ import annotations

import argparse

from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    write_human_reference_sequences,
    write_twobit_sequences,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    human = subparsers.add_parser("human")
    human.add_argument("anchors")
    human.add_argument("twobit")
    human.add_argument("chrom_sizes")
    human.add_argument("output")
    human.add_argument("--target-length", type=int, default=255)

    projected = subparsers.add_parser("projected")
    projected.add_argument("accepted")
    projected.add_argument("twobit")
    projected.add_argument("sequences")
    projected.add_argument("rejected")
    projected.add_argument("--target-length", type=int, default=255)

    args = parser.parse_args(argv)
    if args.command == "human":
        write_human_reference_sequences(
            args.anchors,
            args.twobit,
            args.chrom_sizes,
            args.output,
            target_length=args.target_length,
        )
    else:
        assert args.command == "projected"
        write_twobit_sequences(
            args.accepted,
            args.twobit,
            args.sequences,
            args.rejected,
            target_length=args.target_length,
        )


if __name__ == "__main__":
    main()
