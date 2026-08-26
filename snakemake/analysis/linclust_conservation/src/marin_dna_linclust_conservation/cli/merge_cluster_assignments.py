"""Union complete MMseqs2 cluster partitions from independent Linclust passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna_linclust_conservation.mmseqs import merge_cluster_assignments


def _concatenate_text(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as destination:
        for path in paths:
            text = path.read_text()
            destination.write(text)
            if text and not text.endswith("\n"):
                destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, nargs="+", required=True)
    parser.add_argument("--resources", type=Path, nargs="+", required=True)
    parser.add_argument("--versions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-assignments", type=Path, required=True)
    parser.add_argument("--output-resources", type=Path, required=True)
    parser.add_argument("--output-version", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    assert len(args.assignments) == len(args.resources) == len(args.versions)
    versions = [path.read_text().strip() for path in args.versions]
    assert versions and len(set(versions)) == 1, "ensemble MMseqs2 versions differ"
    receipt = merge_cluster_assignments(
        assignment_paths=args.assignments,
        output_path=args.output_assignments,
    )
    receipt["mmseqs_version"] = versions[0]
    _concatenate_text(args.resources, args.output_resources)
    args.output_version.parent.mkdir(parents=True, exist_ok=True)
    args.output_version.write_text(f"{versions[0]}\n")
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
