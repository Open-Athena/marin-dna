"""Dense local scoring of two complete chromosomes against whole genomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from window_conservation.run import measured


def extract(source: Path, destination: Path, chromosomes: set[str]) -> None:
    found = set()
    keep = False
    with source.open() as src, destination.open("w") as out:
        for line in src:
            if line.startswith(">"):
                name = line[1:].split()[0]
                keep = name in chromosomes
                if keep:
                    found.add(name)
            if keep:
                out.write(line)
    assert found == chromosomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads(Path("config/protocol.json").read_text())
    query = root / "data/query.fa"
    extract(root / "data/human.fa", query, set(protocol["query_scope"]))
    for directory in ["indexes", "scores", "logs", "report"]:
        (root / directory).mkdir(exist_ok=True)
    exe = str(Path("prevalence").resolve())
    listing = root / "data/genomes.list"
    species = sum(bool(line.strip()) for line in listing.read_text().splitlines())
    bits = str(protocol["hash_sample_bits"])
    for k in protocol["k"]:
        index = root / "indexes" / f"k{k}.bin"
        measured(
            [exe, "build-query", str(k), bits, str(listing), str(index), str(query)],
            root / "logs" / f"k{k}-build",
        )
        measured(
            [
                exe,
                "score",
                str(k),
                bits,
                str(species),
                str(protocol["window_bases"]),
                str(index),
                str(query),
                str(root / "scores" / f"k{k}-human.tsv"),
            ],
            root / "logs" / f"k{k}-score",
        )


if __name__ == "__main__":
    main()
