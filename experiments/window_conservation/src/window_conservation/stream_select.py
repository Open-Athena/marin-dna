"""Exact per-genome budget selection using score histograms and disk radix ties."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


def tie_hash(chrom: str, start: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"577:{chrom}:{start}".encode()).digest()[:8], "big"
    )


def radix_cutoff(path: Path, rank: int) -> tuple[int, int]:
    """Select a 1-based hash rank in eight bounded-memory byte passes."""
    assert 1 <= rank <= path.stat().st_size // 8 and path.stat().st_size % 8 == 0
    prefix = 0
    for shift in range(56, -1, -8):
        histogram = [0] * 256
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(2**20), b""):
                for (value,) in struct.iter_unpack("<Q", block):
                    if value >> (shift + 8) == prefix:
                        histogram[(value >> shift) & 255] += 1
        for byte, count in enumerate(histogram):
            if rank <= count:
                prefix = prefix * 256 + byte
                break
            rank -= count
        else:
            raise AssertionError("radix rank absent")
    return prefix, rank


def select(source: Path, output: Path, score_name: str, fraction: float) -> dict:
    assert 0 < fraction <= 1
    output.mkdir(exist_ok=False)
    with source.open() as handle:
        fields = handle.readline().rstrip().split("\t")
    score_index = fields.index(score_name)
    assert fields[:5] == ["species", "chrom", "start", "end", "valid"]
    histograms: dict[str, Counter] = defaultdict(Counter)
    total_rows = 0
    seen_species: set[str] = set()
    previous_species = None
    with source.open() as handle:
        handle.readline()
        for line in handle:
            row = line.rstrip().split("\t")
            if row[0] != previous_species:
                assert row[0] not in seen_species, "input species must be contiguous"
                seen_species.add(row[0])
                previous_species = row[0]
            assert int(row[3]) - int(row[2]) == 100
            total_rows += 1
            if int(row[4]) >= 95:
                value = float(row[score_index])
                assert 0 <= value <= 1
                histograms[row[0]][value] += 1
    plans = {}
    for species, histogram in histograms.items():
        budget = max(1, int(fraction * histogram.total()))
        remaining = budget
        for score, count in sorted(histogram.items(), reverse=True):
            if remaining <= count:
                plans[species] = {
                    "eligible_windows": histogram.total(),
                    "selected_windows": budget,
                    "threshold": score,
                    "ties_needed": remaining,
                    "threshold_ties": count,
                    "distinct_scores": len(histogram),
                }
                break
            remaining -= count
    tie_file = None
    current = None
    try:
        with source.open() as handle:
            handle.readline()
            for line in handle:
                row = line.rstrip().split("\t")
                species = row[0]
                if (
                    int(row[4]) < 95
                    or species not in plans
                    or float(row[score_index]) != plans[species]["threshold"]
                ):
                    continue
                if species != current:
                    if tie_file is not None:
                        tie_file.close()
                    tie_file = (output / f"ties-{species}.bin").open("ab")
                    current = species
                tie_file.write(struct.pack("<Q", tie_hash(row[1], int(row[2]))))
    finally:
        if tie_file is not None:
            tie_file.close()
    for species, plan in plans.items():
        path = output / f"ties-{species}.bin"
        assert path.stat().st_size == plan["threshold_ties"] * 8
        cutoff, equal = radix_cutoff(path, plan["ties_needed"])
        plan["hash_cutoff"] = cutoff
        plan["equal_hash_needed"] = equal
        path.unlink()
    selected = Counter()
    equal_used = Counter()
    bed_stream = None
    bed_species = None
    active = None
    ordinal = Counter()

    def flush() -> None:
        nonlocal active, bed_stream, bed_species
        if active is None:
            return
        species, chrom, start, end, score_sum, bins = active
        ordinal[species] += 1
        if species != bed_species:
            if bed_stream is not None:
                bed_stream.close()
            bed_stream = (output / f"species-{species}.bed").open("w")
            bed_species = species
        mean = score_sum / bins
        assert bed_stream is not None
        bed_stream.write(
            f"{chrom}\t{start}\t{end}\tcandidate_{ordinal[species]}\t{round(mean * 1000)}\t.\t{mean:.8g}\t{bins}\n"
        )
        active = None

    try:
        with source.open() as handle:
            handle.readline()
            for line in handle:
                row = line.rstrip().split("\t")
                species, chrom = row[:2]
                start, end = int(row[2]), int(row[3])
                score = float(row[score_index])
                chosen = False
                if int(row[4]) >= 95 and species in plans:
                    plan = plans[species]
                    chosen = score > plan["threshold"]
                    if score == plan["threshold"]:
                        value = tie_hash(chrom, start)
                        chosen = value < plan["hash_cutoff"]
                        if (
                            value == plan["hash_cutoff"]
                            and equal_used[species] < plan["equal_hash_needed"]
                        ):
                            chosen = True
                            equal_used[species] += 1
                if not chosen:
                    flush()
                    continue
                selected[species] += 1
                if (
                    active is None
                    or active[0] != species
                    or active[1] != chrom
                    or active[3] != start
                ):
                    flush()
                    active = [species, chrom, start, end, score, 1]
                else:
                    active[3] = end
                    active[4] += score
                    active[5] += 1
        flush()
    finally:
        if bed_stream is not None:
            bed_stream.close()
    for species, plan in plans.items():
        assert selected[species] == plan["selected_windows"]
        plan["stretches"] = ordinal[species]
    result = {
        "input_rows": total_rows,
        "fraction": fraction,
        "score": score_name,
        "species": plans,
        "tie_rule": "SHA256(577:chrom:start) first 64 bits, then input order for exact hash collisions; budgets independent per species.",
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", default="any_copy4")
    parser.add_argument("--fraction", type=float, default=0.05)
    args = parser.parse_args()
    print(
        json.dumps(select(args.input, args.output, args.score, args.fraction)),
        flush=True,
    )


if __name__ == "__main__":
    main()
