#!/usr/bin/env python3
"""Compare issue-517 Mendelian specialist metrics with exp232."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pyarrow.parquet as pq


SCORE_TYPE = "minus_llr_avg"
SUBSETS = (
    "missense_variant",
    "splicing",
    "synonymous_variant",
    "5_prime_UTR_variant",
    "tss_proximal",
    "3_prime_UTR_variant",
    "distal",
    "non_coding_transcript_exon_variant",
)


def _read_metrics(path: str) -> dict[str, dict[str, int | float]]:
    table = pq.read_table(
        path,
        columns=["score_type", "subset", "value", "se", "n_groups", "n_rows"],
    )
    columns = {name: table[name].to_pylist() for name in table.column_names}
    result: dict[str, dict[str, int | float]] = {}
    for index in range(table.num_rows):
        if columns["score_type"][index] != SCORE_TYPE:
            continue
        subset = str(columns["subset"][index])
        if subset not in SUBSETS:
            continue
        result[subset] = {
            "value": float(columns["value"][index]),
            "se": float(columns["se"][index]),
            "n_groups": int(columns["n_groups"][index]),
            "n_rows": int(columns["n_rows"][index]),
        }
    if set(result) != set(SUBSETS):
        raise ValueError(f"{path} did not contain exactly the expected subsets")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        nargs=3,
        metavar=("NAME", "CURRENT", "EXP232"),
        required=True,
    )
    args = parser.parse_args()

    arms: dict[str, dict[str, Any]] = {}
    ranked_changes: list[dict[str, Any]] = []
    for arm, current_path, baseline_path in args.arm:
        current = _read_metrics(current_path)
        baseline = _read_metrics(baseline_path)
        rows: dict[str, Any] = {}
        for subset in SUBSETS:
            current_row = current[subset]
            baseline_row = baseline[subset]
            if (
                current_row["n_groups"] != baseline_row["n_groups"]
                or current_row["n_rows"] != baseline_row["n_rows"]
            ):
                raise ValueError(f"support differs for {arm} / {subset}")
            delta = float(current_row["value"]) - float(baseline_row["value"])
            row = {
                "current": current_row,
                "exp232": baseline_row,
                "delta": delta,
            }
            rows[subset] = row
            ranked_changes.append({"arm": arm, "subset": subset, **row})
        arms[arm] = rows

    ranked_changes.sort(key=lambda row: float(row["delta"]), reverse=True)
    json.dump(
        {
            "score_type": SCORE_TYPE,
            "excluded_subset": "mature_miRNA_variant",
            "arms": arms,
            "changes_descending": ranked_changes,
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
