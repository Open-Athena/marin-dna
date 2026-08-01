"""Benchmark batched twoBit sequence extraction against known-good output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    write_twobit_sequences,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accepted", type=Path)
    parser.add_argument("twobit", type=Path)
    parser.add_argument("expected_sequences", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    assert args.accepted.is_file()
    assert args.twobit.is_file()
    assert args.expected_sequences.is_file()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = args.output_dir / "sequences.parquet"
    rejected_path = args.output_dir / "rejected.parquet"

    started = time.monotonic()
    write_twobit_sequences(
        args.accepted,
        args.twobit,
        sequences_path,
        rejected_path,
    )
    elapsed_seconds = time.monotonic() - started

    actual = pl.read_parquet(sequences_path)
    expected = pl.read_parquet(args.expected_sequences)
    assert actual.schema == expected.schema
    non_sequence_columns = [column for column in actual.columns if column != "sequence"]
    assert actual.select(non_sequence_columns).equals(
        expected.select(non_sequence_columns)
    )
    actual_sequences = actual["sequence"]
    expected_sequences = expected["sequence"]
    assert actual_sequences.str.to_uppercase().equals(
        expected_sequences.str.to_uppercase()
    ), "twoBitToFa changed nucleotide content or orientation"
    case_changed_rows = int((actual_sequences != expected_sequences).sum())
    new_rows_with_lowercase = int(actual_sequences.str.contains("[a-z]").sum())
    old_rows_with_lowercase = int(expected_sequences.str.contains("[a-z]").sum())
    rejected = pl.read_parquet(rejected_path)
    assert rejected.is_empty()
    max_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "case_changed_rows": case_changed_rows,
                "elapsed_seconds": elapsed_seconds,
                "max_rss_mib": max_rss_mib,
                "new_rows_with_lowercase": new_rows_with_lowercase,
                "old_rows_with_lowercase": old_rows_with_lowercase,
                "rows": actual.height,
                "sequence_rejections": rejected.height,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
