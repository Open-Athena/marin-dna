"""Check the MMseqs2 synthetic release gate and write a JSON receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna_linclust_conservation.controls import check_release_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = check_release_gate(str(args.assignments), str(args.alignments))
    receipt["mmseqs_version"] = args.mmseqs_version
    receipt["configuration"] = json.loads(args.configuration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
