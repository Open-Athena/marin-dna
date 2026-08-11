"""Regenerate the committed UCSC 2bit size/checksum manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from marin_dna_vertebrate_projection.sequence_sources import build_twobit_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species-manifest",
        type=Path,
        default=Path("config/species_selected.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/twobit_manifest.tsv"),
    )
    args = parser.parse_args()
    frame = build_twobit_manifest(args.species_manifest, args.output)
    print(f"wrote {frame.height} pinned 2bit objects to {args.output}")


if __name__ == "__main__":
    main()
