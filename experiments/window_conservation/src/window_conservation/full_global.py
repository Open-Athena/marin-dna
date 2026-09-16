"""All-window real-genome feasibility trial, including exact BED selection."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from window_conservation.run import measured
from window_conservation.scaling import resource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    trial = args.root / "extension/global3"
    trial.mkdir(exist_ok=True)
    manifest = json.loads((args.root / "data/manifest.json").read_text())
    expected_counts = {
        str(i): sum(length // 100 for length in source["chromosomes"].values())
        for i, source in enumerate(manifest["sources"], 1)
    }
    prior = resource(args.root / "logs/k25-build")["measurements"][0]
    # Conservative simultaneous upper bound: every occurrence creates both a
    # word record and a contribution, plus metadata/output and 8 GiB reserve.
    estimated_scratch = (
        48 * prior["selected_occurrences"]
        + 100 * sum(expected_counts.values())
        + 8 * 2**30
    )
    assert shutil.disk_usage(trial).free > estimated_scratch
    scores = trial / "scores.tsv"
    measured(
        [
            str(Path("partition").resolve()),
            "25",
            "2",
            "100",
            "32",
            str(args.root / "data/genomes.list"),
            str(trial / "scratch"),
            str(scores),
        ],
        args.root / "extension/logs/global3-score",
    )
    count: Counter[str] = Counter()
    compared = 0
    with (
        scores.open() as actual,
        (args.root / "scores/k25-human.tsv").open() as expected,
    ):
        actual.readline()
        expected.readline()
        for line in actual:
            species, row = line.split("\t", 1)
            count[species] += 1
            if species == "1" and row.split("\t", 1)[0] in {"chr1", "chr2"}:
                assert row == expected.readline(), (
                    f"global/pilot score mismatch at row {compared}"
                )
                compared += 1
        assert expected.readline() == ""
    assert count == expected_counts and compared == 4_911_499
    print(
        "global real score parity passed", sum(count.values()), "intervals", flush=True
    )
    measured(
        [
            ".venv/bin/python",
            "-m",
            "window_conservation.stream_select",
            "--input",
            str(scores),
            "--output",
            str(trial / "selection"),
            "--score",
            "any_copy4",
            "--fraction",
            "0.05",
        ],
        args.root / "extension/logs/global3-select",
    )
    summary = {
        "query_parity_rows": compared,
        "all_window_counts": dict(count),
        "score_bytes": scores.stat().st_size,
        "count_and_score": resource(args.root / "extension/logs/global3-score"),
        "selection": resource(args.root / "extension/logs/global3-select"),
        "selection_budget": "Exact 5% independently per species; minimum one eligible interval; k25 any_copy4 at 1/4 sampling.",
        "biological_scope": "The full score universe is three complete genomes. Biological annotations validate human intervals only; this run does not validate selected mouse or armadillo regions.",
    }
    (args.root / "extension/report/global3.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("full three-genome scoring and selection complete", flush=True)


if __name__ == "__main__":
    main()
