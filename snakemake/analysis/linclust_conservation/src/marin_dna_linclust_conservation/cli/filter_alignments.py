"""Keep complete strand-aware alignments for Linclust membership edges."""

from __future__ import annotations

import argparse
from pathlib import Path

from marin_dna_linclust_conservation.mmseqs import filter_cluster_alignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    filtered = filter_cluster_alignments(
        assignments_path=args.assignments,
        alignments_paths=args.alignments,
        output_path=args.output,
    )
    print(f"retained {filtered.height} strand-aware cluster alignments")


if __name__ == "__main__":
    main()
