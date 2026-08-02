"""Frozen constants and validation for feature 1662's untouched-test transfer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from common import ISSUE, sha256_file, write_json

DATASET = "marin-dna/evals_complex_traits"
DATASET_REVISION = "22f86a89c65cb8f3007ac3cc2739f40efefa4340"
TEST_OBJECT = "test.parquet"
EXPECTED_BYTES = 1_036_640
EXPECTED_SHA256 = "4bc355e8a39ce310d792d5fb0293ef01a7fc6306eef6415dae01f81133520ab6"
EXPECTED_ROWS = 10_000
EXPECTED_GROUPS = 1_000
EXPECTED_POSITIVES = 1_000
FEATURE_ID = 1_662
BLOCK_INDEX = 18
EXPECTED_SUBSET_COUNTS = {
    "3_prime_UTR_variant": (290, 29),
    "5_prime_UTR_variant": (190, 19),
    "distal": (5_770, 577),
    "missense_variant": (2_040, 204),
    "non_coding_transcript_exon_variant": (380, 38),
    "splicing": (110, 11),
    "synonymous_variant": (160, 16),
    "tss_proximal": (1_060, 106),
}


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    assert np.all((0 <= values) & (values <= 1))
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(monotone, 0, 1)
    return adjusted


def validate_test_panel(path: Path) -> dict[str, Any]:
    assert path.is_file() and path.stat().st_size == EXPECTED_BYTES
    assert sha256_file(path) == EXPECTED_SHA256
    frame = pl.read_parquet(path)
    assert frame.height == EXPECTED_ROWS
    assert frame["label"].dtype == pl.Boolean
    assert int(frame["label"].sum()) == EXPECTED_POSITIVES
    assert frame["match_group"].n_unique() == EXPECTED_GROUPS
    groups = frame.group_by("match_group").agg(
        pl.len().alias("rows"),
        pl.col("label").sum().alias("positives"),
        pl.col("subset").n_unique().alias("subsets"),
    )
    assert groups.height == EXPECTED_GROUPS
    assert set(groups["rows"].unique()) == {10}
    assert set(groups["positives"].unique()) == {1}
    assert set(groups["subsets"].unique()) == {1}
    observed = {
        row["subset"]: (row["rows"], row["positives"])
        for row in frame.group_by("subset")
        .agg(pl.len().alias("rows"), pl.col("label").sum().alias("positives"))
        .to_dicts()
    }
    assert observed == EXPECTED_SUBSET_COUNTS
    assert all(rows == 10 * positives for rows, positives in observed.values())
    return {
        "issue": ISSUE,
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "object": TEST_OBJECT,
        "bytes": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
        "rows": EXPECTED_ROWS,
        "match_groups": EXPECTED_GROUPS,
        "positives": EXPECTED_POSITIVES,
        "subsets": {
            subset: {"rows": rows, "positives": positives}
            for subset, (rows, positives) in sorted(observed.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, validate_test_panel(args.panel))


if __name__ == "__main__":
    main()
