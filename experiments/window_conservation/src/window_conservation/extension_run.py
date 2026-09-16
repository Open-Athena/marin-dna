"""Score nested species panels with fixed-rate and bottom-hash samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from window_conservation.run import measured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["rate", "bottom", "all"], default="all")
    args = parser.parse_args()
    root = args.root / "extension"
    for mode, bits, bottom in [("rate", 2, 0), ("bottom", 0, 32)]:
        if args.mode != "all" and mode != args.mode:
            continue
        measured(
            [
                str(Path("compact").resolve()),
                "25",
                str(bits),
                str(root / "data/genomes.list"),
                str(root / "data/query.fa"),
                str(root / "scores" / mode),
                "3,6,10",
                "100",
                "0",
                str(bottom),
            ],
            root / "logs" / f"{mode}-panels",
        )


if __name__ == "__main__":
    main()
