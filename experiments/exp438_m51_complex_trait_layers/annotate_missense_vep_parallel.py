"""Parallel prefetcher for the resumable official-Ensembl VEP annotation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl

from annotate_missense_vep import EXPECTED_MISSENSE_ROWS, variant_input
from annotate_missense_vep_resumable import (
    BATCH_SIZE,
    atomic_write_json,
    request_with_recursive_split,
    run,
)

MAX_WORKERS = 4


def fetch_checkpoint(path: Path, batch: list[dict[str, Any]]) -> tuple[Path, str]:
    if path.is_file():
        result = json.loads(path.read_text())
        assert isinstance(result, list) and len(result) == len(batch)
        return path, "cached"
    result = request_with_recursive_split([variant_input(row) for row in batch])
    atomic_write_json(path, result)
    return path, "fetched"


def prefetch(candidate_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    candidates = pl.read_parquet(candidate_path)
    missense = candidates.filter(pl.col("subset") == "missense_variant").sort(
        "mean_abs_delta", descending=True
    )
    assert missense.height == EXPECTED_MISSENSE_ROWS
    rows = missense.select("panel_row", "chrom", "pos", "ref", "alt").to_dicts()
    jobs: list[tuple[Path, list[dict[str, Any]]]] = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        path = batches_dir / f"batch-{start:04d}-{start + len(batch):04d}.json"
        jobs.append((path, batch))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_checkpoint, path, batch): path for path, batch in jobs
        }
        complete = 0
        for future in as_completed(futures):
            path, status = future.result()
            complete += 1
            print(
                f"VEP batches {complete}/{len(jobs)} ({status}: {path.name})",
                flush=True,
            )
    run(candidate_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prefetch(args.candidate_table, args.output_dir)


if __name__ == "__main__":
    main()
