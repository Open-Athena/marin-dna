"""Sequential resource-profiled construction and unary scoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def measured(command: list[str], prefix: Path) -> None:
    started = time.time()
    with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
        result = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(prefix.with_suffix(".time")), *command],
            stdout=out, stderr=err, check=False,
        )
    prefix.with_suffix(".receipt.json").write_text(json.dumps(
        {"command": command, "start_epoch": started, "end_epoch": time.time(), "returncode": result.returncode},
        indent=2,
    ) + "\n")
    if result.returncode:
        raise RuntimeError(f"failed: {command}; see {prefix}.stderr")
    print(prefix.name, "complete", round(time.time() - started, 3), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[17, 21, 25])
    args = parser.parse_args()
    root = args.root
    (root / "indexes").mkdir(exist_ok=True)
    (root / "scores").mkdir(exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    exe = str(Path("prevalence").resolve())
    for k in args.k:
        index = root / "indexes" / f"k{k}.bin"
        measured([exe, "build", str(k), "6", str(root / "data/genomes.list"), str(index)], logs / f"k{k}-build")
        output = root / "scores" / f"k{k}"
        measured([exe, "score-list", str(k), "6", "3", "4096", str(index),
                  str(root / "data/genomes.list"), str(output)], logs / f"k{k}-score")
        for species in ["human", "mouse", "armadillo"]:
            (output / f"{species}.tsv").rename(root / "scores" / f"k{k}-{species}.tsv")


if __name__ == "__main__":
    main()
