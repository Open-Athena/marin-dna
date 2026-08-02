"""Validate and manifest the pinned official complex-traits training panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from common import ISSUE, sha256_file, write_json

DATASET_ID = "marin-dna/evals_complex_traits"
DATASET_REVISION = "22f86a89c65cb8f3007ac3cc2739f40efefa4340"
TRAIN_PARQUET_SHA256 = (
    "d65fcee2740317451c41e5df6d4dd52cddf33847578b7565075e38d74d9865e3"
)
EXPECTED_ROWS = 11_630
EXPECTED_GROUPS = 1_163
EXPECTED_ROWS_PER_GROUP = 10
EXPECTED_POSITIVES_PER_GROUP = 1
EXPECTED_SUBSET_COUNTS = {
    "3_prime_UTR_variant": (490, 49),
    "5_prime_UTR_variant": (370, 37),
    "distal": (6_160, 616),
    "missense_variant": (2_500, 250),
    "non_coding_transcript_exon_variant": (370, 37),
    "splicing": (190, 19),
    "synonymous_variant": (170, 17),
    "tss_proximal": (1_380, 138),
}
NUCLEOTIDES = frozenset("ACGT")


def validate_panel(path: Path) -> dict[str, Any]:
    assert path.is_file(), path
    observed_sha256 = sha256_file(path)
    assert observed_sha256 == TRAIN_PARQUET_SHA256
    frame = pl.read_parquet(path)
    required = {"chrom", "pos", "ref", "alt", "label", "subset", "match_group"}
    assert required <= set(frame.columns), required - set(frame.columns)
    assert frame.height == EXPECTED_ROWS
    assert (
        frame.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == frame.height
    )
    assert frame.filter(pl.col("pos") < 1).is_empty()
    assert frame.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert frame.filter(
        ~pl.col("ref").is_in(sorted(NUCLEOTIDES))
        | ~pl.col("alt").is_in(sorted(NUCLEOTIDES))
    ).is_empty()
    assert frame.select(pl.col("label").is_null().any()).item() is False
    assert frame.select(pl.col("label").unique().sort()).to_series().to_list() == [
        False,
        True,
    ]

    groups = frame.group_by("match_group").agg(
        pl.len().alias("rows"),
        pl.col("label").sum().alias("positives"),
        pl.col("subset").n_unique().alias("subsets"),
    )
    assert groups.height == EXPECTED_GROUPS
    assert groups.filter(
        (pl.col("rows") != EXPECTED_ROWS_PER_GROUP)
        | (pl.col("positives") != EXPECTED_POSITIVES_PER_GROUP)
        | (pl.col("subsets") != 1)
    ).is_empty()

    subset_rows = frame.group_by("subset").agg(
        pl.len().alias("rows"), pl.col("label").sum().alias("positives")
    )
    observed_subsets = {
        row["subset"]: (row["rows"], row["positives"])
        for row in subset_rows.iter_rows(named=True)
    }
    assert observed_subsets == EXPECTED_SUBSET_COUNTS
    return {
        "issue": ISSUE,
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": "train",
        "panel_sha256": observed_sha256,
        "bytes": path.stat().st_size,
        "row_count": frame.height,
        "match_groups": groups.height,
        "prevalence": float(frame["label"].mean()),
        "subset_counts": {
            subset: {"rows": rows, "positives": positives}
            for subset, (rows, positives) in sorted(observed_subsets.items())
        },
        "invariants": {
            "coordinate_system": "VCF-style 1-based positions at input boundary",
            "rows_per_match_group": EXPECTED_ROWS_PER_GROUP,
            "positives_per_match_group": EXPECTED_POSITIVES_PER_GROUP,
            "one_subset_per_match_group": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_panel(args.panel)
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
