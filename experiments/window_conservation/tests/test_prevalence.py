from __future__ import annotations

import csv
import json
import random
import struct
import subprocess
from collections import Counter
from pathlib import Path

import pytest

MASK = 2**64 - 1


def mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def keys(sequence: str, k: int, bits: int) -> list[tuple[int, int]]:
    output = []
    for end in range(k, len(sequence) + 1):
        word = sequence[end-k:end].upper()
        if set(word) - set("ACGT"):
            continue
        reverse = word.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        code = min(int("".join(str("ACGT".index(c)) for c in x), 4) for x in [word, reverse])
        key = mix(code ^ 577)
        if key % 2**bits == 0:
            output.append((end, key))
    return output


@pytest.fixture(scope="session")
def executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("native") / "prevalence"
    subprocess.run(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "src/window_conservation/prevalence.cpp", "-o", str(path)], check=True)
    return path


@pytest.mark.parametrize("k,bits", [(3, 0), (9, 2), (17, 3), (31, 0)])
def test_species_counts_and_unary_scores(executable: Path, tmp_path: Path, k: int, bits: int) -> None:
    rng = random.Random(577)
    ancestor = "".join(rng.choices("ACGT", k=512))
    sequences = [ancestor, ancestor[:200] + "N" * 10 + ancestor[210:], ancestor.lower()]
    records = [[s[:256], s[256:]] for s in sequences]
    records[0].append(ancestor[:256])  # Copy number must not inflate species support.
    paths = []
    reference = []
    for si, seqs in enumerate(records):
        path = tmp_path / f"s{si}.fa"
        path.write_text("".join(f">c{i}\n{s[:90]}\n{s[90:]}\n" for i, s in enumerate(seqs)))
        paths.append(path)
        reference.append(Counter(key for s in seqs for _, key in keys(s, k, bits)))
    species_list = tmp_path / "species.list"
    species_list.write_text("\n".join(map(str, paths)) + "\n")
    index = tmp_path / "index.bin"
    receipt = json.loads(subprocess.check_output([str(executable), "build", str(k), str(bits), str(species_list), str(index)]))
    observed = {key: (count, copies) for key, count, copies, _ in struct.iter_unpack("<QIHH", index.read_bytes())}
    expected = {key: (sum(key in r for r in reference), max(r[key] for r in reference))
                for key in set().union(*(set(r) for r in reference))}
    assert observed == expected
    assert receipt["unique_keys"] == len(expected)
    assert receipt["bases"] == sum(len(s) for seqs in records for s in seqs)
    output = tmp_path / "scores.tsv"
    subprocess.run([str(executable), "score", str(k), str(bits), "3", "64", str(index), str(paths[1]), str(output)], check=True)
    rows = list(csv.DictReader(output.open(), delimiter="\t"))
    assert len(rows) == 8
    for row in rows:
        sequence = records[1][int(row["chrom"][1:])]
        start, end = int(row["start"]), int(row["end"])
        selected = {key for position, key in keys(sequence, k, bits) if start < position <= end}
        assert int(row["seeds"]) == len(selected)
        denom = max(1, len(selected))
        for field, function in {
            "any": lambda n: n >= 2,
            "both": lambda n: n >= 3,
            "breadth": lambda n: (n-1)/2,
        }.items():
            expected_score = sum(function(expected[key][0]) for key in selected) / denom
            expected_filtered = sum(function(expected[key][0]) for key in selected if expected[key][1] <= 4) / denom
            assert float(row[field]) == pytest.approx(expected_score, abs=1e-6)
            assert float(row[field + "_copy4"]) == pytest.approx(expected_filtered, abs=1e-6)


def test_reverse_complement_and_ambiguity(executable: Path, tmp_path: Path) -> None:
    sequence = "ACGTTAGCGATCGATTTAGC"
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    a, b = tmp_path / "a.fa", tmp_path / "b.fa"
    a.write_text(f">a\n{sequence}NNN{sequence}\n")
    b.write_text(f">b\n{reverse}\n")
    paths = tmp_path / "list"
    paths.write_text(f"{a}\n{b}\n")
    index = tmp_path / "index"
    subprocess.run([str(executable), "build", "9", "0", str(paths), str(index)], check=True)
    entries = list(struct.iter_unpack("<QIHH", index.read_bytes()))
    assert entries and all(n == 2 for _, n, _, _ in entries)
    assert all(copies >= 2 for _, _, copies, _ in entries)


def test_invalid_parameters_fail(executable: Path, tmp_path: Path) -> None:
    result = subprocess.run([str(executable), "build", "32", "6", str(tmp_path / "missing"), str(tmp_path / "out")], capture_output=True, check=False)
    assert result.returncode != 0
    assert not (tmp_path / "out").exists()


def test_batch_scoring_and_input_limit(executable: Path, tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        path = tmp_path / f"species{i}.fa"
        path.write_text(">chr1\n" + "ACGT" * 100 + "\n")
        paths.append(path)
    listing = tmp_path / "list"
    listing.write_text("\n".join(map(str, paths)) + "\n")
    index = tmp_path / "index"
    result = json.loads(subprocess.check_output([str(executable), "build", "9", "0", str(listing), str(index), "128"]))
    assert result["bases"] == 3 * 128
    out = tmp_path / "scores"
    subprocess.run([str(executable), "score-list", "9", "0", "3", "64", str(index), str(listing), str(out), "128"], check=True)
    assert len(list(out.glob("*.tsv"))) == 3
    for path in out.glob("*.tsv"):
        rows = list(csv.DictReader(path.open(), delimiter="\t"))
        assert len(rows) == 2
        assert all(float(row["both"]) == 1 for row in rows)
        assert all(float(row["both_copy4"]) == 0 for row in rows)
