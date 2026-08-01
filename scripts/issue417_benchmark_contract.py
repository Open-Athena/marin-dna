"""Benchmark bounded-memory projection contraction against known-good outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    write_contract_outputs,
    write_contract_outputs_for_alignment,
)


def _assert_parquet_equal(actual_path: Path, expected_path: Path) -> int:
    actual = pl.read_parquet(actual_path)
    expected = pl.read_parquet(expected_path)
    assert actual.schema == expected.schema
    assert actual.equals(expected)
    return actual.height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fragments", type=Path)
    parser.add_argument("expected_accepted", type=Path)
    parser.add_argument("expected_rejected", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--alignment-name")
    args = parser.parse_args()

    assert args.fragments.is_file()
    assert args.expected_accepted.is_file()
    assert args.expected_rejected.is_file()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted.parquet"
    rejected_path = args.output_dir / "rejected.parquet"

    started = time.monotonic()
    if args.alignment_name is None:
        write_contract_outputs(
            args.fragments,
            accepted_path,
            rejected_path,
            target_length=255,
            pre_resize_min_length=128,
            pre_resize_max_length=512,
        )
    else:
        write_contract_outputs_for_alignment(
            args.fragments,
            args.alignment_name,
            accepted_path,
            rejected_path,
            target_length=255,
            pre_resize_min_length=128,
            pre_resize_max_length=512,
        )
    elapsed_seconds = time.monotonic() - started
    accepted_rows = _assert_parquet_equal(accepted_path, args.expected_accepted)
    rejected_rows = _assert_parquet_equal(rejected_path, args.expected_rejected)
    max_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "accepted_rows": accepted_rows,
                "alignment_name": args.alignment_name,
                "rejected_rows": rejected_rows,
                "elapsed_seconds": elapsed_seconds,
                "max_rss_mib": max_rss_mib,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
