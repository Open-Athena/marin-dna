"""Correlate effective epochs with home-specialist AUPRC for issue #517."""

from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path


INPUT_DIR = Path(".agents/artifacts/issue-517/evaluation")
HOME_ROWS_PATH = INPUT_DIR / "issue517_historical_diagonals_home_rows.csv"
EPOCHS_PATH = INPUT_DIR / "issue517_historical_effective_epochs.csv"
POINTS_OUTPUT_PATH = INPUT_DIR / "issue517_epoch_performance_home_points.csv"
CORRELATIONS_OUTPUT_PATH = INPUT_DIR / "issue517_epoch_performance_correlations.csv"

EXPERIMENT_NAMES = {
    "issue232_v4": "exp232",
    "issue517_annotation_first": "annotation_first",
    "issue517_gpn_uniform": "gpn_uniform",
    "issue517_phylop_uniform": "phylop_uniform",
}
ARM_NAMES = {
    "CDS": "CDS",
    "3-prime UTR": "3′ UTR",
    "ncRNA exon": "ncRNA exon",
    "TSS / 5-prime UTR": "TSS / 5′ UTR",
    "Enhancer": "Enhancer",
}


def correlation(x: list[float], y: list[float]) -> float:
    assert len(x) == len(y)
    assert len(x) >= 3
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    numerator = sum(a * b for a, b in zip(x_centered, y_centered))
    denominator = math.sqrt(
        sum(value**2 for value in x_centered)
        * sum(value**2 for value in y_centered)
    )
    assert denominator > 0
    return numerator / denominator


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def exact_permutation_p(
    x: list[float],
    y: list[float],
    *,
    use_ranks: bool,
) -> float:
    if use_ranks:
        x = rank(x)
        y = rank(y)
    observed = abs(correlation(x, y))
    permuted = (
        abs(correlation(x, list(permutation)))
        for permutation in itertools.permutations(y)
    )
    exceedances = sum(value >= observed - 1e-12 for value in permuted)
    return exceedances / math.factorial(len(y))


def main() -> None:
    epoch_index: dict[tuple[str, str], dict[str, str]] = {}
    with EPOCHS_PATH.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            if row["experiment"] not in EXPERIMENT_NAMES:
                continue
            if row["arm"] not in ARM_NAMES:
                continue
            key = (
                EXPERIMENT_NAMES[row["experiment"]],
                ARM_NAMES[row["arm"]],
            )
            assert key not in epoch_index
            epoch_index[key] = row

    points: list[dict[str, str]] = []
    with HOME_ROWS_PATH.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            epoch = epoch_index[(row["experiment"], row["home_arm"])]
            points.append(
                {
                    **row,
                    "train_rows": epoch["train_rows"],
                    "sequence_presentations": epoch["sequence_presentations"],
                    "effective_epochs": epoch["effective_row_epochs"],
                }
            )
    assert len(points) == 32

    points_by_subset: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for point in points:
        points_by_subset[point["subset"]].append(point)
        assert float(point["effective_epochs"]) > 0
        assert 0 <= float(point["home_auprc"]) <= 1
    assert len(points_by_subset) == 8
    assert all(len(rows) == 4 for rows in points_by_subset.values())
    assert {point["experiment"] for point in points} == set(EXPERIMENT_NAMES.values())

    correlation_rows: list[dict[str, float | int | str | bool]] = []
    for subset, rows in points_by_subset.items():
        x = [float(row["effective_epochs"]) for row in rows]
        y = [float(row["home_auprc"]) for row in rows]
        pearson_r = correlation(x, y)
        spearman_rho = correlation(rank(x), rank(y))
        leave_one_out = [
            correlation(x[:index] + x[index + 1 :], y[:index] + y[index + 1 :])
            for index in range(len(rows))
        ]
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        slope = sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y)
        ) / sum((x_value - x_mean) ** 2 for x_value in x)
        correlation_rows.append(
            {
                "subset": subset,
                "home_arm": rows[0]["home_arm"],
                "n_experiments": len(rows),
                "pearson_r": f"{pearson_r:.6f}",
                "pearson_exact_p": f"{exact_permutation_p(x, y, use_ranks=False):.6f}",
                "spearman_rho": f"{spearman_rho:.6f}",
                "spearman_exact_p": f"{exact_permutation_p(x, y, use_ranks=True):.6f}",
                "auprc_per_additional_epoch": f"{slope:.6f}",
                "leave_one_out_pearson_min": f"{min(leave_one_out):.6f}",
                "leave_one_out_pearson_max": f"{max(leave_one_out):.6f}",
                "leave_one_out_sign_stable": bool(
                    all(value > 0 for value in leave_one_out)
                    or all(value < 0 for value in leave_one_out)
                ),
            }
        )

    assert len(correlation_rows) == 8
    for row in correlation_rows:
        assert -1 <= float(row["pearson_r"]) <= 1
        assert -1 <= float(row["spearman_rho"]) <= 1
        assert 0 <= float(row["pearson_exact_p"]) <= 1
        assert 0 <= float(row["spearman_exact_p"]) <= 1

    with POINTS_OUTPUT_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=points[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(points)
    with CORRELATIONS_OUTPUT_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=correlation_rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(correlation_rows)


if __name__ == "__main__":
    main()
