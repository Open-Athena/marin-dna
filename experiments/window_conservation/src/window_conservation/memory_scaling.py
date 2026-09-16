"""Matched compact and disk-partitioned global resource comparisons."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path

from window_conservation.run import measured
from window_conservation.scaling import resource


def audit_scores(root: Path, run: Path, species: int, windows: int, mode: str) -> int:
    """Compare every score against the archived-in-place baseline synthetic run."""
    reference = root / "scaling" / f"s{species}-w{windows}-r0/scores"
    if mode == "compact":
        for i in range(species):
            name = f"species-{i:04d}.tsv"
            assert filecmp.cmp(
                reference / name, run / "scores" / str(species) / name, shallow=False
            )
        return species * windows
    count, current = 0, 0
    expected = None
    try:
        with (run / "scores.tsv").open() as actual:
            actual.readline()
            for line in actual:
                name, row = line.split("\t", 1)
                identifier = int(name)
                if identifier != current:
                    if expected is not None:
                        assert expected.readline() == ""
                        expected.close()
                    expected = (reference / f"species-{identifier - 1:04d}.tsv").open()
                    expected.readline()
                    current = identifier
                assert expected is not None and row == expected.readline()
                count += 1
            assert expected is not None and expected.readline() == ""
    finally:
        if expected is not None:
            expected.close()
    assert count == species * windows
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root / "extension/memory"
    root.mkdir(exist_ok=True)
    results = []
    for repeat in range(3):
        arms = [
            (species, windows, bits, mode)
            for species, windows in [
                (125, 2048),
                (1000, 2048),
                (250, 1024),
                (250, 8192),
            ]
            for bits in [2, 4]
            for mode in ["compact", "partition"]
        ]
        if repeat % 2:
            arms.reverse()
        for species, windows, bits, mode in arms:
            name = f"{mode}-s{species}-w{windows}-b{bits}-r{repeat}"
            run = root / name
            run.mkdir(exist_ok=True)
            paths = [
                args.root / "scaling/synthetic" / f"species-{i:04d}.fa"
                for i in range(species)
            ]
            assert all(path.exists() for path in paths)
            listing = run / "list"
            listing.write_text("\n".join(map(str, paths)) + "\n")
            if mode == "compact":
                command = [
                    str(Path("compact").resolve()),
                    "25",
                    str(bits),
                    str(listing),
                    "-",
                    str(run / "scores"),
                    str(species),
                    "100",
                    str(windows * 100),
                ]
            else:
                command = [
                    str(Path("partition").resolve()),
                    "25",
                    str(bits),
                    "100",
                    "32",
                    str(listing),
                    str(run / "scratch"),
                    str(run / "scores.tsv"),
                    str(windows * 100),
                ]
            measured(command, run / "measure")
            row = resource(run / "measure")
            receipt = row["measurements"][0]
            assert receipt["bases"] == species * windows * 100
            observed_windows = (
                receipt["windows"]
                if mode == "partition"
                else sum(
                    r["windows"] for r in row["measurements"] if r["stage"] == "score"
                )
            )
            assert observed_windows == species * windows
            row["baseline_parity_rows"] = (
                audit_scores(args.root, run, species, windows, mode)
                if bits == 2
                else None
            )
            results.append(
                {
                    "mode": mode,
                    "species": species,
                    "windows_per_species": windows,
                    "bits": bits,
                    "repeat": repeat,
                    **row,
                }
            )
            (root / "measurements.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            # Recomputable synthetic tables; preserve source, resource receipts, and audits.
            if mode == "compact":
                shutil.rmtree(run / "scores")
            else:
                (run / "scores.tsv").unlink()
    print("memory matrix complete", len(results), flush=True)


if __name__ == "__main__":
    main()
