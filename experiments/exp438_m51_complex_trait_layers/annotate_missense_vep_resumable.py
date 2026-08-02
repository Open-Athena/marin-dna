"""Resumable official-Ensembl VEP annotation for feature 1662 missense rows."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from annotate_missense_vep import (
    API_ROOT,
    EXPECTED_MISSENSE_ROWS,
    VEP_OPTIONS,
    flatten_result,
    variant_input,
)
from common import ISSUE, write_json

BATCH_SIZE = 50
REQUEST_TIMEOUT_SECONDS = 45
MAX_ATTEMPTS = 2


def request_batch(variants: list[str]) -> list[dict[str, Any]]:
    assert 0 < len(variants) <= BATCH_SIZE
    endpoint = (
        f"{API_ROOT}/vep/homo_sapiens/region?{urllib.parse.urlencode(VEP_OPTIONS)}"
    )
    payload = json.dumps({"variants": variants}).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                assert response.status == 200
                result = json.loads(response.read())
                assert isinstance(result, list) and len(result) == len(variants)
                return result
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code not in {
                429,
                500,
                502,
                503,
                504,
            }:
                raise
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def request_with_recursive_split(variants: list[str]) -> list[dict[str, Any]]:
    try:
        return request_batch(variants)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        if len(variants) == 1:
            raise
        midpoint = len(variants) // 2
        print(
            f"Splitting slow batch {len(variants)} -> {midpoint}+{len(variants) - midpoint}",
            flush=True,
        )
        return request_with_recursive_split(
            variants[:midpoint]
        ) + request_with_recursive_split(variants[midpoint:])


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def get_release_metadata() -> Any:
    request = urllib.request.Request(
        f"{API_ROOT}/info/data?", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        assert response.status == 200
        return json.loads(response.read())


def annotate_resumably(frame: pl.DataFrame, batches_dir: Path) -> list[dict[str, Any]]:
    batches_dir.mkdir(parents=True, exist_ok=True)
    rows = frame.select("panel_row", "chrom", "pos", "ref", "alt").to_dicts()
    all_results: list[dict[str, Any]] = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        path = batches_dir / f"batch-{start:04d}-{start + len(batch):04d}.json"
        if path.is_file():
            results = json.loads(path.read_text())
            assert isinstance(results, list) and len(results) == len(batch)
            status = "cached"
        else:
            results = request_with_recursive_split(
                [variant_input(row) for row in batch]
            )
            atomic_write_json(path, results)
            status = "fetched"
        all_results.extend(results)
        print(f"VEP {start + len(batch)}/{len(rows)} ({status})", flush=True)
        time.sleep(0.2)
    return all_results


def run(candidate_path: Path, output_dir: Path) -> None:
    assert candidate_path.is_file()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pl.read_parquet(candidate_path)
    missense = candidates.filter(pl.col("subset") == "missense_variant").sort(
        "mean_abs_delta", descending=True
    )
    assert missense.height == EXPECTED_MISSENSE_ROWS
    assert missense["panel_row"].n_unique() == missense.height
    raw_results = annotate_resumably(missense, output_dir / "batches")
    assert len(raw_results) == missense.height
    annotation = pl.DataFrame([flatten_result(result) for result in raw_results])
    assert annotation.height == missense.height
    assert annotation["panel_row"].n_unique() == annotation.height
    assert set(annotation["assembly_name"].unique()) == {"GRCh38"}
    merged = missense.join(annotation, on="panel_row", how="inner", validate="1:1")
    assert merged.height == missense.height

    atomic_write_json(output_dir / "vep_raw.json", raw_results)
    annotation.write_parquet(output_dir / "vep_annotations.parquet", compression="zstd")
    merged.write_parquet(output_dir / "missense_annotated.parquet", compression="zstd")
    write_json(
        output_dir / "results.json",
        {
            "issue": ISSUE,
            "analysis_status": "post_hoc_descriptive",
            "annotated_at": datetime.now(UTC).isoformat(),
            "api_root": API_ROOT,
            "api_release": get_release_metadata(),
            "vep_options": VEP_OPTIONS,
            "batch_size": BATCH_SIZE,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "rows": merged.height,
            "assembly_name": "GRCh38",
            "fields_with_values": {
                column: int(merged[column].is_not_null().sum())
                for column in (
                    "sift_score",
                    "polyphen_score",
                    "blosum62",
                    "alphamissense_score",
                    "clinical_significance",
                )
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.candidate_table, args.output_dir)


if __name__ == "__main__":
    main()
