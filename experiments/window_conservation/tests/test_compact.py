"""Compare compact counting against the separate baseline implementation."""

import random
import subprocess
from pathlib import Path

import pytest


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
