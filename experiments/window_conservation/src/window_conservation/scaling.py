"""Factorial resource checks; synthetic streams carry no biological evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from window_conservation.run import measured


def generate(
    root: Path, species: int = 1000, windows: int = 8192, width: int = 100
) -> list[Path]:
    data = root / "synthetic"
    data.mkdir(exist_ok=True)
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    paths = []
    for index in range(species):
        path = data / f"species-{index:04d}.fa"
        if not path.exists():
            rng = np.random.default_rng(577 + index)
            with path.open("wb") as handle:
                handle.write(b">synthetic\n")
                # Bound generation memory; native prefix limits are enforced per base.
                for offset in range(0, windows * width, 2**16):
                    count = min(2**16, windows * width - offset)
                    handle.write(
                        alphabet[rng.integers(0, 4, count, dtype=np.uint8)].tobytes()
                        + b"\n"
                    )
        paths.append(path)
    (data / "generator.json").write_text(
        json.dumps(
            {
                "seed": 577,
                "species": species,
                "windows": windows,
                "width": width,
                "alphabet": "uniform ACGT",
                "biological_validation": False,
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def resource(prefix: Path) -> dict:
    receipt = json.loads(prefix.with_suffix(".receipt.json").read_text())
    text = prefix.with_suffix(".time").read_text()
    rss = next(
        int(line.split(":")[-1])
        for line in text.splitlines()
        if "Maximum resident set size" in line
    )
    rows = [
        json.loads(line)
        for line in prefix.with_suffix(".stdout").read_text().splitlines()
        if line.startswith("{")
    ]
    return {
        "wall_seconds": receipt["end_epoch"] - receipt["start_epoch"],
        "max_rss_kib": rss,
        "measurements": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    root = args.root / "scaling"
    root.mkdir(exist_ok=True)
    paths = generate(root)
    shapes = sorted(
        set(
            [(s, 2048) for s in [125, 250, 500, 1000]]
            + [(250, w) for w in [1024, 2048, 4096, 8192]]
        )
    )
    exe = str(Path("prevalence").resolve())
    results = []
    # Alternate traversal order across repetitions to reduce a monotone warm-cache confound.
    for repeat in range(args.repetitions):
        for species, windows in shapes if repeat % 2 == 0 else shapes[::-1]:
            run = root / f"s{species}-w{windows}-r{repeat}"
            run.mkdir(exist_ok=True)
            listing = run / "species.list"
            listing.write_text("\n".join(map(str, paths[:species])) + "\n")
            index = run / "index.bin"
            limit = str(windows * 100)
            measured(
                [exe, "build", str(args.k), "2", str(listing), str(index), limit],
                run / "build",
            )
            measured(
                [
                    exe,
                    "score-list",
                    str(args.k),
                    "2",
                    str(species),
                    "100",
                    str(index),
                    str(listing),
                    str(run / "scores"),
                    limit,
                ],
                run / "score",
            )
            build, score = resource(run / "build"), resource(run / "score")
            assert build["measurements"][0]["bases"] == species * windows * 100
            assert (
                sum(row["windows"] for row in score["measurements"])
                == species * windows
            )
            row = {
                "species": species,
                "windows_per_species": windows,
                "windows": species * windows,
                "k": args.k,
                "repeat": repeat,
                "build": build,
                "score": score,
                "total_seconds": build["wall_seconds"] + score["wall_seconds"],
                "max_rss_kib": max(build["max_rss_kib"], score["max_rss_kib"]),
            }
            results.append(row)
            (root / "measurements.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in [
                            "species",
                            "windows_per_species",
                            "repeat",
                            "total_seconds",
                            "max_rss_kib",
                        ]
                    }
                ),
                flush=True,
            )
            index.unlink()  # Recomputable task-owned scratch; keep sizes and operation counts.


if __name__ == "__main__":
    main()
