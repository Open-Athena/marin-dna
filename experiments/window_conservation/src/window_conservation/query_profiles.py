"""Measure the standalone query index footprint of each sampling scheme."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from window_conservation.run import measured
from window_conservation.scaling import resource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-dir", default="extension")
    parser.add_argument("--exclude-lowercase", action="store_true")
    parser.add_argument("--maximum-repeat", type=float, default=1.0)
    args = parser.parse_args()
    root = args.root / args.experiment_dir
    output = root / "profiles"
    output.mkdir(exist_ok=True)
    rows = []
    for sample, bits, bottom in [
        ("rate4", 2, 0),
        ("rate8", 3, 0),
        ("rate16", 4, 0),
        ("bottom16", 0, 16),
        ("bottom32", 0, 32),
    ]:
        prefix = output / sample
        measured(
            [
                str(Path("compact").resolve()),
                "25",
                str(bits),
                "-",
                str(root / "data/query.fa"),
                str(output / sample),
                "3",
                "100",
                "0",
                str(bottom),
                str(int(args.exclude_lowercase)),
                str(args.maximum_repeat),
            ],
            prefix,
        )
        rows.append({"sample": sample, **resource(prefix)})
        print("query profile", sample, flush=True)
    (root / "report/query-profiles.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
