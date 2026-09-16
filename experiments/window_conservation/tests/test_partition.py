"""Disk partitioning must preserve global scores, including copies and tails."""

from __future__ import annotations

import csv
import random
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def partition_exe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("partition-build") / "partition"
    subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DWINDOW_CHUNK_SIZE=2",
            "src/window_conservation/partition.cpp",
            "-o",
            str(target),
        ],
        check=True,
    )
    return target


@pytest.mark.parametrize("parts,bits", [(1, 0), (4, 0), (32, 2), (32, 4)])
def test_global_score_parity(
    partition_exe: Path, tmp_path: Path, parts: int, bits: int
) -> None:
    baseline = tmp_path / "baseline"
    subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++17",
            "src/window_conservation/prevalence.cpp",
            "-o",
            str(baseline),
        ],
        check=True,
    )
    rng = random.Random(577)
    sequence = "".join(rng.choices("ACGT", k=350))
    unique = "".join(rng.choices("ACGT", k=240))
    paths = []
    for i in range(3):
        path = tmp_path / f"s{i}.fa"
        reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        path.write_text(
            f">first\n{sequence * (i + 4)}\n>partial\n{sequence[:90]}\n>last\n{reverse[:100]}NNN{reverse.lower()}\n>single\n{unique}\n"
        )
        paths.append(path)
    listing = tmp_path / "list"
    listing.write_text("\n".join(map(str, paths)) + "\n")
    output = tmp_path / "output.tsv"
    subprocess.run(
        [
            str(partition_exe),
            "17",
            str(bits),
            "100",
            str(parts),
            str(listing),
            str(tmp_path / "scratch"),
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(baseline),
            "build",
            "17",
            str(bits),
            str(listing),
            str(tmp_path / "index"),
        ],
        check=True,
        capture_output=True,
    )
    rows = list(csv.DictReader(output.open(), delimiter="\t"))
    for number, path in enumerate(paths, 1):
        expected = tmp_path / f"s{number}.tsv"
        subprocess.run(
            [
                str(baseline),
                "score",
                "17",
                str(bits),
                "3",
                "100",
                str(tmp_path / "index"),
                str(path),
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        reference = list(csv.DictReader(expected.open(), delimiter="\t"))
        selected = [row for row in rows if row["species"] == str(number)]
        assert len(reference) == len(selected)
        for left, right in zip(selected, reference, strict=True):
            assert left["chrom"] == right["chrom"]
            for field in right.keys() - {"chrom"}:
                assert float(left[field]) == pytest.approx(
                    float(right[field]), abs=1e-6
                )
    assert not (tmp_path / "scratch").exists()


def test_partition_mask_matches_manually_hardmasked_words(
    partition_exe: Path, tmp_path: Path
) -> None:
    rng = random.Random(581)
    sequence = "".join(rng.choices("ACGT", k=400))
    masked = sequence[:50] + sequence[50:70].lower() + sequence[70:110] + sequence[110:131].lower() + sequence[131:]
    results = []
    for mode in ["soft", "hard"]:
        paths = []
        for i, text in enumerate([masked, sequence, sequence.lower()]):
            if mode == "hard":
                text = "".join("N" if c.islower() else c for c in text)
            path = tmp_path / f"{mode}-{i}.fa"
            path.write_text(f">chr1\n{text}\n")
            paths.append(path)
        listing = tmp_path / f"{mode}-list"
        listing.write_text("\n".join(map(str, paths)) + "\n")
        output = tmp_path / f"{mode}.tsv"
        subprocess.run([
            str(partition_exe), "17", "0", "100", "4", str(listing),
            str(tmp_path / f"scratch-{mode}"), str(output), "0",
            "1" if mode == "soft" else "0", "0.2" if mode == "soft" else "1",
        ], check=True, capture_output=True)
        results.append(list(csv.DictReader(output.open(), delimiter="\t")))
    for actual, expected in zip(*results, strict=True):
        excluded = float(actual["repeat"]) > 0.2
        for field in ["seeds", "any", "both", "breadth", "any_copy4", "both_copy4", "breadth_copy4"]:
            assert float(actual[field]) == pytest.approx(0 if excluded else float(expected[field]))
    assert float(results[0][0]["repeat"]) == 0.2 and int(results[0][0]["seeds"]) > 0
