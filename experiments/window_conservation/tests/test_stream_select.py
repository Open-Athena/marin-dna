import csv
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from window_conservation.evaluate import select_indices
from window_conservation.stream_select import (
    radix_cutoff,
    score_cutoff,
    select,
    tie_hash,
)


def test_weighted_score_radix() -> None:
    counts = Counter({0.0: 10, 1 / 3: 5, 0.75: 2, 1.0: 1})
    reference = sorted(counts.elements(), reverse=True)
    for rank, expected in enumerate(reference, 1):
        score, needed = score_cutoff(counts, rank)
        assert score == expected
        assert needed == rank - sum(value > expected for value in reference)


def test_radix_ties_and_extremes(tmp_path: Path) -> None:
    values = [0, 1, 1, 256, 2**63 + 1, 2**64 - 1]
    path = tmp_path / "ties"
    path.write_bytes(b"".join(struct.pack("<Q", value) for value in values[::-1]))
    for rank, expected in enumerate(values, 1):
        observed, equal = radix_cutoff(path, rank)
        assert observed == expected
        assert equal == rank - sum(value < expected for value in values)


def test_streaming_selection_matches_partition_selector(tmp_path: Path) -> None:
    source = tmp_path / "scores.tsv"
    with source.open("w") as handle:
        handle.write("species\tchrom\tstart\tend\tvalid\tany_copy4\n")
        for species in [1, 2]:
            for i in range(80):
                handle.write(
                    f"{species}\tchr1\t{i * 100}\t{(i + 1) * 100}\t{0 if i == 20 else 100}\t{(i % 4) / 4 if species == 1 else 0}\n"
                )
    report = select(source, tmp_path / "out", "any_copy4", 0.25)
    for species in [1, 2]:
        ids = np.array([i for i in range(80) if i != 20])
        scores = np.array([(i % 4) / 4 if species == 1 else 0 for i in ids])
        tie = np.array([tie_hash("chr1", i * 100) for i in ids], dtype=np.uint64)
        expected = set(ids[select_indices(scores, tie, 19)])
        observed = set()
        with (tmp_path / "out" / f"species-{species}.bed").open() as handle:
            for row in csv.reader(handle, delimiter="\t"):
                observed.update(range(int(row[1]) // 100, int(row[2]) // 100))
        assert observed == expected
        assert report["species"][str(species)]["selected_windows"] == 19


def test_repeat_cutoff_is_inclusive_and_applies_to_every_pass(tmp_path: Path) -> None:
    source = tmp_path / "repeat.tsv"
    source.write_text(
        "species\tchrom\tstart\tend\tvalid\trepeat\tany_copy4\n"
        "1\tchr1\t0\t100\t100\t0.21\t1\n"
        "1\tchr1\t100\t200\t100\t0.20\t0.5\n"
        "1\tchr1\t200\t300\t100\t0.00\t0.5\n"
        "1\tchr1\t300\t400\t100\t0.21\t0.5\n"
        "1\tchr1\t400\t500\t94\t0.00\t1\n"
        "2\tchr1\t0\t100\t100\t1.00\t1\n"
    )
    for fraction in [0.5, 1.0]:
        output = tmp_path / str(fraction)
        result = select(source, output, "any_copy4", fraction, 0.2)
        assert result["species"]["1"]["eligible_windows"] == 2
        assert "2" not in result["species"]
        observed = set()
        for row in csv.reader((output / "species-1.bed").open(), delimiter="\t"):
            observed.update(range(int(row[1]), int(row[2]), 100))
        assert observed <= {100, 200}
        assert len(observed) == int(2 * fraction)
