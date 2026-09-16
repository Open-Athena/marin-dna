import csv
import struct
from pathlib import Path

import numpy as np

from window_conservation.evaluate import select_indices
from window_conservation.stream_select import radix_cutoff, select, tie_hash


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
