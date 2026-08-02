from __future__ import annotations

import math

import polars as pl

from build_panel import (
    FOCAL_INDEX,
    FRACTION_COLUMNS,
    FUNCTIONAL_CLASSES,
    REFERENCE_CLASSES,
    WINDOW_BP,
    deterministic_balanced_sample,
    extract_sequence,
    pure_class_candidates,
    sequence_metrics,
    stable_hash,
)


def _label_row(
    name: str,
    *,
    label: str,
    pure_fraction: str,
    start: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "chrom": "1",
        "start": start,
        "end": start + WINDOW_BP,
        "label": label,
    }
    row.update({column: 0.0 for column in FRACTION_COLUMNS})
    row[pure_fraction] = 1.0
    return row


def _all_class_rows(*, repeats: int = 1) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        for index, (reference_class, fraction_column) in enumerate(
            FUNCTIONAL_CLASSES.items()
        ):
            rows.append(
                _label_row(
                    f"{reference_class}-{repeat}",
                    label=reference_class,
                    pure_fraction=fraction_column,
                    start=(repeat * len(REFERENCE_CLASSES) + index) * WINDOW_BP,
                )
            )
        rows.append(
            _label_row(
                f"intron-{repeat}",
                label="background",
                pure_fraction="intron_frac",
                start=(repeat * len(REFERENCE_CLASSES) + 5) * WINDOW_BP,
            )
        )
        rows.append(
            _label_row(
                f"intergenic-{repeat}",
                label="background",
                pure_fraction="intergenic_frac",
                start=(repeat * len(REFERENCE_CLASSES) + 6) * WINDOW_BP,
            )
        )
    return pl.DataFrame(rows)


def test_pure_class_candidates_maps_disjoint_classes() -> None:
    labels = _all_class_rows()
    candidates = pure_class_candidates(labels)

    assert candidates.height == len(REFERENCE_CLASSES)
    assert set(candidates["reference_class"]) == set(REFERENCE_CLASSES)
    assert candidates.filter(pl.col("label") == "background").height == 2


def test_deterministic_balanced_sample_uses_smallest_hashes() -> None:
    candidates = pure_class_candidates(_all_class_rows(repeats=3))

    first = deterministic_balanced_sample(candidates, samples_per_class=2)
    second = deterministic_balanced_sample(candidates.reverse(), samples_per_class=2)

    assert first.equals(second)
    assert first["panel_row"].to_list() == list(range(first.height))
    for reference_class in REFERENCE_CLASSES:
        observed = set(
            first.filter(pl.col("reference_class") == reference_class)["name"]
        )
        eligible = candidates.filter(pl.col("reference_class") == reference_class)[
            "name"
        ].to_list()
        expected = set(
            sorted(eligible, key=lambda name: stable_hash(reference_class, name))[:2]
        )
        assert observed == expected


def test_extract_sequence_uses_zero_based_half_open_coordinates() -> None:
    genome = {"1": "A" * 10 + "C" * WINDOW_BP + "G" * 10}

    sequence = extract_sequence(genome, chrom="1", start=10, end=10 + WINDOW_BP)

    assert sequence == "C" * WINDOW_BP
    assert sequence[FOCAL_INDEX] == "C"


def test_sequence_metrics() -> None:
    sequence = "A" * 100 + "C" * 55 + "G" * 50 + "T" * 50

    metrics = sequence_metrics(sequence)

    assert metrics["gc_fraction"] == 105 / WINDOW_BP
    assert metrics["cpg_count"] == 1
    assert metrics["maximum_homopolymer"] == 100
    assert metrics["n_fraction"] == 0
    expected_entropy = -sum(
        (count / WINDOW_BP) * math.log2(count / WINDOW_BP)
        for count in (100, 55, 50, 50)
    )
    assert math.isclose(metrics["sequence_entropy"], expected_entropy)
