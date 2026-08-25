"""Update the smoke run's peak temporary-disk byte counter."""

from __future__ import annotations

import argparse
from pathlib import Path

from marin_dna_linclust_conservation.resources import record_peak_temporary_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    peak = record_peak_temporary_bytes(
        directory=args.directory,
        output_path=args.output,
    )
    print(peak)


if __name__ == "__main__":
    main()
