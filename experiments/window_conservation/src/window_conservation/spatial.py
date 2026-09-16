"""Mechanistic localization checks with planted tracts and negative controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from window_conservation.run import measured


def dna(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


def mutate(
    sequence: str, substitution: float, indel: float, rng: np.random.Generator
) -> str:
    """Per ancestral base: independent deletion, substitution, and insertion."""
    output = []
    for base in sequence:
        if rng.random() < indel / 2:
            continue
        if rng.random() < substitution:
            base = str(rng.choice([x for x in "ACGT" if x != base]))
        output.append(base)
        if rng.random() < indel / 2:
            output.append(str(rng.choice(list("ACGT"))))
    return "".join(output)


def generate(folder: Path) -> list[dict]:
    rng = np.random.default_rng(577)
    handles = [(folder / f"species{i}.fa").open("w") for i in range(3)]
    cases = []
    try:
        for length in [50, 100, 150, 300]:
            for substitution in [0, 0.05, 0.10, 0.20]:
                for indel in [0, 0.01]:
                    for control in ["unique", "eight_copy", "shuffled"]:
                        for repeat in range(20):
                            chrom = f"case{len(cases)}"
                            start = int(rng.integers(750, 850))
                            tract = dna(rng, length)
                            query = (
                                dna(rng, start)
                                + tract
                                + dna(rng, 2000 - start - length)
                            )
                            handles[0].write(f">{chrom}\n{query}\n")
                            for species in [1, 2]:
                                target = mutate(tract, substitution, indel, rng)
                                if control == "shuffled":
                                    target = "".join(rng.permutation(list(target)))
                                sequence = (
                                    dna(rng, 800)
                                    + target
                                    + dna(rng, 2000 - 800 - len(target))
                                )
                                handles[species].write(f">{chrom}\n{sequence}\n")
                                if control == "eight_copy":
                                    for copy in range(7):
                                        handles[species].write(
                                            f">{chrom}_copy{copy}\n{target}\n"
                                        )
                            cases.append(
                                {
                                    "chrom": chrom,
                                    "start": start,
                                    "end": start + length,
                                    "tract_length": length,
                                    "substitution": substitution,
                                    "indel": indel,
                                    "control": control,
                                    "replicate": repeat,
                                }
                            )
    finally:
        for handle in handles:
            handle.close()
    (folder / "genomes.list").write_text(
        "".join(str(folder / f"species{i}.fa") + "\n" for i in range(3))
    )
    (folder / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return cases


def summarize(path: Path, cases: list[dict], sample: str) -> list[dict]:
    with path.open() as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))
    by_chrom: dict[str, list[dict]] = {}
    for row in table:
        by_chrom.setdefault(row["chrom"], []).append(row)
    output = []
    for case in cases:
        rows = by_chrom[case["chrom"]]
        assert len(rows) == 20
        starts = np.array([int(row["start"]) for row in rows])
        ends = starts + 100
        overlap = np.maximum(
            0, np.minimum(ends, case["end"]) - np.maximum(starts, case["start"])
        )
        near = (ends > case["start"] - 100) & (starts < case["end"] + 100)
        for field in ["any", "any_copy4"]:
            score = np.array([float(row[field]) for row in rows])
            positive = score > 0
            peak = int(np.argmax(score))
            output.append(
                {
                    **case,
                    "sample": sample,
                    "score": field,
                    "peak_score": float(score[peak]),
                    "positive_detection": bool(np.any(positive & (overlap > 0))),
                    "peak_bin_overlaps_tract": bool(
                        score[peak] > 0 and overlap[peak] > 0
                    ),
                    "planted_base_coverage": float(
                        overlap[positive].sum() / case["tract_length"]
                    ),
                    "positive_bin_tract_fraction": float(
                        overlap[positive].sum() / max(1, 100 * positive.sum())
                    ),
                    "far_neutral_positive_fraction": float(positive[~near].mean()),
                    "selected_bins": int(positive.sum()),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root / "extension/spatial"
    root.mkdir(exist_ok=True)
    cases = generate(root)
    for mode, bits, bottom in [("rate", 2, 0), ("bottom", 0, 32)]:
        measured(
            [
                str(Path("compact").resolve()),
                "25",
                str(bits),
                str(root / "genomes.list"),
                str(root / "species0.fa"),
                str(root / mode),
                "3",
                "100",
                "0",
                str(bottom),
            ],
            args.root / "extension/logs" / f"spatial-{mode}",
        )
    observations = []
    for sample, filename in [
        ("rate4", "rate/3/human.tsv"),
        ("rate8", "rate/3/bits3.tsv"),
        ("rate16", "rate/3/bits4.tsv"),
        ("bottom16", "bottom/3/bottom16.tsv"),
        ("bottom32", "bottom/3/human.tsv"),
    ]:
        observations.extend(summarize(root / filename, cases, sample))
    with (root / "observations.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    grouped: dict[tuple, list[dict]] = {}
    fields = ["sample", "score", "tract_length", "substitution", "indel", "control"]
    measures = [
        "peak_score",
        "positive_detection",
        "peak_bin_overlaps_tract",
        "planted_base_coverage",
        "positive_bin_tract_fraction",
        "far_neutral_positive_fraction",
        "selected_bins",
    ]
    for row in observations:
        grouped.setdefault(tuple(row[key] for key in fields), []).append(row)
    summary = [
        {
            **dict(zip(fields, key, strict=True)),
            "replicates": len(rows),
            **{
                measure: float(np.mean([row[measure] for row in rows]))
                for measure in measures
            },
        }
        for key, rows in grouped.items()
    ]
    (args.root / "extension/report/spatial.json").write_text(
        json.dumps(
            {
                "design": "1920 independent query tracts in 2000bp contexts, two mutated support genomes; 20 replicates per cell. Positive means any retained shared word, not a tuned biological threshold. Shuffled regions are negative controls, not conserved positives.",
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print("spatial controls complete", len(cases), "cases", flush=True)


if __name__ == "__main__":
    main()
