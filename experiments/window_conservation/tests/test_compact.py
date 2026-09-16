"""Compare compact counting against the separate baseline implementation."""

import csv
import random
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from test_prevalence import keys


@pytest.fixture(scope="module")
def executables(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    folder = tmp_path_factory.mktemp("compact-build")
    result = []
    for name in ["prevalence", "compact"]:
        target = folder / name
        subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"src/window_conservation/{name}.cpp",
                "-o",
                str(target),
            ],
            check=True,
        )
        result.append(target)
    return result[0], result[1]


@pytest.mark.parametrize("bits", [0, 2, 4])
@pytest.mark.parametrize("restricted", [False, True])
def test_exact_scores(
    tmp_path: Path, executables: tuple[Path, Path], bits: int, restricted: bool
) -> None:
    baseline, compact = executables
    rng = random.Random(577)
    shared = "".join(rng.choices("ACGT", k=2000))
    reverse = shared.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    paths = []
    for species in range(3):
        path = tmp_path / f"s{species}.fa"
        background = "".join(rng.choices("ACGTacgtN", k=30000))
        path.write_text(
            f">chr1\n{background}{shared}\n>chr2\n{reverse * (species + 4)}NN\n"
        )
        paths.append(path)
    listing = tmp_path / "genomes.list"
    listing.write_text("\n".join(map(str, paths)) + "\n")
    index = tmp_path / "baseline.bin"
    command = [
        str(baseline),
        "build-query" if restricted else "build",
        "17",
        str(bits),
        str(listing),
        str(index),
    ]
    if restricted:
        command.append(str(paths[0]))
    subprocess.run(command, check=True, capture_output=True)
    out = tmp_path / "compact"
    subprocess.run(
        [
            str(compact),
            "17",
            str(bits),
            str(listing),
            str(paths[0]) if restricted else "-",
            str(out),
            "1,3",
            "100",
        ],
        check=True,
        capture_output=True,
    )
    for path in paths[:1] if restricted else paths:
        expected = tmp_path / f"{path.stem}.tsv"
        subprocess.run(
            [
                str(baseline),
                "score",
                "17",
                str(bits),
                "3",
                "100",
                str(index),
                str(path),
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        actual = out / "3" / ("human.tsv" if restricted else f"{path.stem}.tsv")
        assert actual.read_bytes() == expected.read_bytes()


def test_bottom_sketch_counts_full_genomes(
    tmp_path: Path, executables: tuple[Path, Path]
) -> None:
    _, compact = executables
    rng = random.Random(578)
    first = "".join(rng.choices("ACGT", k=400))
    other = (
        "".join(rng.choices("ACGT", k=77)) + first[40:180] + "N" * 23 + first[:100] * 5
    )
    sequences = [first, other, first.lower()]
    listing = tmp_path / "list"
    paths = []
    for i, sequence in enumerate(sequences):
        path = tmp_path / f"s{i}.fa"
        path.write_text(f">chr1\n{sequence}\n")
        paths.append(path)
    listing.write_text("\n".join(map(str, paths)) + "\n")
    out = tmp_path / "out"
    subprocess.run(
        [
            str(compact),
            "17",
            "0",
            str(listing),
            str(paths[0]),
            str(out),
            "3",
            "100",
            "0",
            "32",
        ],
        check=True,
        capture_output=True,
    )
    counts = [
        Counter(key for _, key in keys(sequence, 17, 0)) for sequence in sequences
    ]
    for n, filename in [(32, "human.tsv"), (16, "bottom16.tsv")]:
        rows = list(csv.DictReader((out / "3" / filename).open(), delimiter="\t"))
        assert len(rows) == 4
        for row in rows:
            chosen = sorted(
                {
                    key
                    for end, key in keys(first, 17, 0)
                    if int(row["start"]) < end <= int(row["end"])
                }
            )[:n]
            assert int(row["seeds"]) == len(chosen)
            for field, condition in [
                ("any", lambda c: c >= 2),
                ("both", lambda c: c >= 3),
                ("breadth", lambda c: (c - 1) / 2),
            ]:
                for suffix in ["", "_copy4"]:
                    expected = sum(
                        condition(sum(key in count for count in counts))
                        for key in chosen
                        if not suffix or max(count[key] for count in counts) <= 4
                    ) / len(chosen)
                    assert float(row[field + suffix]) == pytest.approx(
                        expected, abs=1e-6
                    )


@pytest.mark.parametrize("bottom,bits", [(0, 0), (0, 2), (32, 0)])
def test_repeat_mask_and_window_cutoff(
    tmp_path: Path, executables: tuple[Path, Path], bottom: int, bits: int
) -> None:
    _, compact = executables
    rng = random.Random(580)
    raw = "".join(rng.choices("ACGT", k=400))
    query = (
        raw[:40] + raw[40:60].lower() + raw[60:140] + raw[140:161].lower() + raw[161:]
    )
    sequences = [query, raw, raw.lower() * 6]
    paths = []
    for i, sequence in enumerate(sequences):
        path = tmp_path / f"s{i}.fa"
        path.write_text(f">chr1\n{sequence}\n")
        paths.append(path)
    listing = tmp_path / "list"
    listing.write_text("\n".join(map(str, paths)) + "\n")
    out = tmp_path / "out"
    subprocess.run(
        [
            str(compact),
            "17",
            str(bits),
            str(listing),
            str(paths[0]),
            str(out),
            "3",
            "100",
            "0",
            str(bottom),
            "1",
            "0.2",
        ],
        check=True,
        capture_output=True,
    )
    masked = ["".join("N" if c.islower() else c for c in s) for s in sequences]
    counts = [Counter(key for _, key in keys(s, 17, bits)) for s in masked]
    for size, filename in (
        [(32, "human.tsv"), (16, "bottom16.tsv")] if bottom else [(0, "human.tsv")]
    ):
        rows = list(csv.DictReader((out / "3" / filename).open(), delimiter="\t"))
        for index, row in enumerate(rows):
            chosen = sorted(
                {
                    key
                    for end, key in keys(masked[0], 17, bits)
                    if index * 100 < end <= (index + 1) * 100
                }
            )
            if size:
                chosen = chosen[:size]
            if index == 1:
                chosen = []  # 21% repeat; exactly 20% in bin zero remains eligible.
            assert int(row["seeds"]) == len(chosen)
            for field, condition in [
                ("any", lambda n: n >= 2),
                ("both", lambda n: n >= 3),
                ("breadth", lambda n: (n - 1) / 2),
            ]:
                expected = sum(
                    condition(sum(key in count for count in counts)) for key in chosen
                ) / max(1, len(chosen))
                assert float(row[field]) == pytest.approx(expected, abs=1e-6)
                assert float(row[field + "_copy4"]) == pytest.approx(expected, abs=1e-6)
            assert float(row["both"]) == 0  # Entire lowercase third genome is absent.
        assert int(rows[0]["seeds"]) > 0
