#!/usr/bin/env python3
"""Validate issue #417 JSONL.zst publication artifacts before HF upload."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import yaml
import zstandard as zstd


@dataclass(frozen=True)
class ShardResult:
    cohort: str
    split: str
    index: int
    path: str
    compressed_bytes: int
    uncompressed_bytes: int
    rows: int
    sha256: str


def _validate_record(
    raw: bytes,
    *,
    expected_columns: frozenset[str],
    split: str,
    validation_chrom: str,
    target_length: int,
) -> None:
    record = json.loads(raw)
    assert set(record) == expected_columns
    assert isinstance(record["sequence"], str)
    assert len(record["sequence"]) == target_length
    assert record["augmentation"] in {"+", "-"}
    if split == "train":
        assert record["source_chrom"] != validation_chrom
    else:
        assert record["source_chrom"] == validation_chrom
        assert record["augmentation"] == "+"


def _validate_shard(
    task: tuple[
        Path,
        str,
        str,
        int,
        frozenset[str],
        str,
        int,
    ],
) -> ShardResult:
    (
        path,
        cohort,
        split,
        index,
        expected_columns,
        validation_chrom,
        target_length,
    ) = task
    digest = hashlib.sha256()
    decompressor = zstd.ZstdDecompressor().decompressobj()
    first_line: bytes | None = None
    last_line: bytes | None = None
    tail = b""
    rows = 0
    uncompressed_bytes = 0

    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
            decoded = decompressor.decompress(chunk)
            uncompressed_bytes += len(decoded)
            parts = (tail + decoded).split(b"\n")
            tail = parts.pop()
            if parts:
                if first_line is None:
                    first_line = parts[0]
                last_line = parts[-1]
                rows += len(parts)
        decoded = decompressor.flush()
        uncompressed_bytes += len(decoded)
        parts = (tail + decoded).split(b"\n")
        tail = parts.pop()
        if parts:
            if first_line is None:
                first_line = parts[0]
            last_line = parts[-1]
            rows += len(parts)

    assert decompressor.eof, f"truncated zstd frame: {path}"
    assert tail == b"", f"JSONL file lacks terminal newline: {path}"
    assert first_line is not None and last_line is not None, f"empty shard: {path}"
    _validate_record(
        first_line,
        expected_columns=expected_columns,
        split=split,
        validation_chrom=validation_chrom,
        target_length=target_length,
    )
    _validate_record(
        last_line,
        expected_columns=expected_columns,
        split=split,
        validation_chrom=validation_chrom,
        target_length=target_length,
    )
    return ShardResult(
        cohort=cohort,
        split=split,
        index=index,
        path=str(path),
        compressed_bytes=path.stat().st_size,
        uncompressed_bytes=uncompressed_bytes,
        rows=rows,
        sha256=digest.hexdigest(),
    )


def validate_artifacts(
    artifact_dir: Path,
    source_datasets_dir: Path,
    output_path: Path,
    *,
    config_path: Path,
    pipeline_commit: str,
    workers: int,
) -> dict[str, object]:
    """Validate the complete artifact tree and write a machine-readable manifest."""
    assert len(pipeline_commit) == 40
    assert workers > 0
    config = yaml.safe_load(config_path.read_text())
    cohorts = list(config["region_cohorts"])
    train_shards = int(config["publication_train_shards"])
    validation_shards = int(config["publication_validation_shards"])
    validation_chrom = str(config["validation_chrom"])
    target_length = int(config["target_length"])
    assert len(cohorts) == len(set(cohorts))
    assert train_shards > 0 and validation_shards > 0

    expected_files: set[Path] = set()
    shard_tasks: list[tuple[Path, str, str, int, frozenset[str], str, int]] = []
    cohort_metadata: dict[str, dict[str, object]] = {}
    forbidden_card_text = [
        "bolinas-dna",
        "loss weight",
        "0.01",
        "train.parquet",
        "validation.parquet",
    ]

    for cohort in cohorts:
        card = artifact_dir / cohort / "README.md"
        expected_files.add(card.relative_to(artifact_dir))
        card_text = card.read_text()
        assert f"# `marin-dna/vertebrate-v1-{cohort}`" in card_text
        assert f"blob/{pipeline_commit}/" in card_text
        assert "path: data/train/*.jsonl.zst" in card_text
        assert "path: data/validation/*.jsonl.zst" in card_text
        for forbidden in forbidden_card_text:
            assert forbidden not in card_text

        train_source = source_datasets_dir / cohort / "train.parquet"
        validation_source = source_datasets_dir / cohort / "validation.parquet"
        train_schema = pl.read_parquet_schema(train_source)
        validation_schema = pl.read_parquet_schema(validation_source)
        assert train_schema == validation_schema
        assert {"sequence", "augmentation", "source_chrom"} <= set(train_schema)
        expected_columns = frozenset(train_schema)
        source_rows = {
            "train": int(
                pl.scan_parquet(train_source)
                .select(pl.len())
                .collect(engine="streaming")
                .item()
            ),
            "validation": int(
                pl.scan_parquet(validation_source)
                .select(pl.len())
                .collect(engine="streaming")
                .item()
            ),
        }
        assert source_rows["train"] > 0
        assert source_rows["validation"] > 0
        cohort_metadata[cohort] = {
            "source_rows": source_rows,
            "schema": {column: str(dtype) for column, dtype in train_schema.items()},
        }

        for split, count in [
            ("train", train_shards),
            ("validation", validation_shards),
        ]:
            for index in range(count):
                path = (
                    artifact_dir
                    / cohort
                    / "data"
                    / split
                    / f"shard_{index:04d}.jsonl.zst"
                )
                expected_files.add(path.relative_to(artifact_dir))
                shard_tasks.append(
                    (
                        path,
                        cohort,
                        split,
                        index,
                        expected_columns,
                        validation_chrom,
                        target_length,
                    )
                )

    actual_files = {
        path.relative_to(artifact_dir)
        for path in artifact_dir.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files, {
        "missing": sorted(str(path) for path in expected_files - actual_files),
        "unexpected": sorted(str(path) for path in actual_files - expected_files),
    }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        shard_results = list(executor.map(_validate_shard, shard_tasks))
    shard_results.sort(key=lambda result: (result.cohort, result.split, result.index))

    manifest_cohorts: dict[str, object] = {}
    for cohort in cohorts:
        split_manifests: dict[str, object] = {}
        metadata = cohort_metadata[cohort]
        source_rows = metadata["source_rows"]
        assert isinstance(source_rows, dict)
        for split in ["train", "validation"]:
            results = [
                result
                for result in shard_results
                if result.cohort == cohort and result.split == split
            ]
            row_counts = [result.rows for result in results]
            assert sum(row_counts) == source_rows[split]
            assert max(row_counts) - min(row_counts) <= 1
            split_manifests[split] = {
                "rows": sum(row_counts),
                "compressed_bytes": sum(result.compressed_bytes for result in results),
                "uncompressed_bytes": sum(
                    result.uncompressed_bytes for result in results
                ),
                "shards": [asdict(result) for result in results],
            }
        manifest_cohorts[cohort] = {
            "schema": metadata["schema"],
            "splits": split_manifests,
        }

    manifest: dict[str, object] = {
        "pipeline_commit": pipeline_commit,
        "artifact_format": "JSONL.zst",
        "validation_chrom": validation_chrom,
        "target_length": target_length,
        "cohorts": manifest_cohorts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-datasets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("snakemake/vertebrate_projection_dataset/config/config.yaml"),
    )
    parser.add_argument("--pipeline-commit", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = validate_artifacts(
        args.artifact_dir,
        args.source_datasets_dir,
        args.output,
        config_path=args.config,
        pipeline_commit=args.pipeline_commit,
        workers=args.workers,
    )
    cohorts = manifest["cohorts"]
    assert isinstance(cohorts, dict)
    print(f"validated {len(cohorts)} cohorts: {args.output}")


if __name__ == "__main__":
    main()
