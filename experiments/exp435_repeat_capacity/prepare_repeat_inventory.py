"""Materialize the annotation-only hg38 RepeatMasker inventory for issue 435.

UCSC database coordinates are already 0-based, half-open. Chromosome-name
normalization is the only coordinate-boundary conversion performed here.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from common import (
    ISSUE,
    PRIMARY_CHROMS,
    RMSK_URL,
    RUN_ID,
    assert_commit,
    sha256_file,
    write_json,
)

RMSK_SCHEMA: Mapping[str, pl.DataType] = {
    "bin": pl.Int64,
    "sw_score": pl.Int64,
    "milli_div": pl.Int64,
    "milli_del": pl.Int64,
    "milli_ins": pl.Int64,
    "geno_name": pl.String,
    "geno_start": pl.Int64,
    "geno_end": pl.Int64,
    "geno_left": pl.Int64,
    "strand": pl.String,
    "repeat_name": pl.String,
    "repeat_class": pl.String,
    "repeat_family": pl.String,
    "repeat_start": pl.Int64,
    "repeat_end": pl.Int64,
    "repeat_left": pl.Int64,
    "record_id": pl.Int64,
}


def normalize_chrom(value: str) -> str:
    assert value.startswith("chr")
    return value.removeprefix("chr")


def read_fai(path: Path) -> dict[str, int]:
    assert path.is_file()
    lengths: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            assert len(fields) >= 2
            raw_chrom = fields[0]
            chrom = (
                normalize_chrom(raw_chrom) if raw_chrom.startswith("chr") else raw_chrom
            )
            if chrom not in PRIMARY_CHROMS:
                continue
            length = int(fields[1])
            assert chrom not in lengths and length > 0
            lengths[chrom] = length
    assert set(lengths) == set(PRIMARY_CHROMS)
    return lengths


def read_repeatmasker(
    path: Path, chrom_lengths: Mapping[str, int]
) -> tuple[pl.DataFrame, int]:
    assert path.is_file() and set(chrom_lengths) == set(PRIMARY_CHROMS)
    source = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        schema=RMSK_SCHEMA,
        quote_char=None,
    )
    source_rows = source.height
    allowed = [f"chr{chrom}" for chrom in PRIMARY_CHROMS]
    annotations = (
        source.filter(pl.col("geno_name").is_in(allowed))
        .select(
            pl.col("geno_name").str.strip_prefix("chr").alias("chrom"),
            pl.col("geno_start").cast(pl.Int64).alias("start0"),
            pl.col("geno_end").cast(pl.Int64).alias("end0"),
            pl.col("sw_score").cast(pl.Int32),
            pl.col("milli_div").cast(pl.Int32),
            pl.col("strand"),
            pl.col("repeat_name"),
            pl.col("repeat_class"),
            pl.col("repeat_family"),
            pl.col("record_id"),
        )
        .with_row_index("annotation_id")
        .with_columns(pl.col("annotation_id").cast(pl.UInt32))
        .sort("chrom", "start0", "end0", "annotation_id")
    )
    assert source_rows >= annotations.height > 0
    assert set(annotations["chrom"].unique()) <= set(PRIMARY_CHROMS)
    assert annotations.filter(pl.col("start0") < 0).is_empty()
    assert annotations.filter(pl.col("end0") <= pl.col("start0")).is_empty()
    assert (
        annotations.select(
            pl.any_horizontal(
                pl.col("repeat_name").is_null() | (pl.col("repeat_name") == ""),
                pl.col("repeat_class").is_null() | (pl.col("repeat_class") == ""),
                pl.col("repeat_family").is_null() | (pl.col("repeat_family") == ""),
            ).any()
        ).item()
        is False
    )
    length_frame = pl.DataFrame(
        {"chrom": list(chrom_lengths), "chrom_length": list(chrom_lengths.values())}
    )
    bounded = annotations.join(length_frame, on="chrom", how="left", validate="m:1")
    assert bounded["chrom_length"].null_count() == 0
    assert bounded.filter(pl.col("end0") > pl.col("chrom_length")).is_empty()
    return annotations, source_rows


def union_length(starts: np.ndarray, ends: np.ndarray) -> tuple[int, int]:
    assert starts.ndim == ends.ndim == 1 and starts.size == ends.size > 0
    assert np.all(starts[:-1] <= starts[1:])
    assert np.all(starts >= 0) and np.all(ends > starts)
    cumulative_end = np.maximum.accumulate(ends)
    is_new = np.concatenate([np.array([True]), starts[1:] > cumulative_end[:-1]])
    segment_starts = starts[is_new]
    first_indices = np.flatnonzero(is_new)
    last_indices = np.concatenate([first_indices[1:] - 1, [starts.size - 1]])
    segment_ends = cumulative_end[last_indices]
    assert segment_starts.size == segment_ends.size
    lengths = segment_ends - segment_starts
    assert np.all(lengths > 0)
    return int(lengths.sum()), int(lengths.size)


def build_chrom_inventory(
    annotations: pl.DataFrame, chrom_lengths: Mapping[str, int]
) -> pl.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    by_chrom = {
        frame["chrom"][0]: frame
        for frame in annotations.partition_by("chrom", maintain_order=True)
    }
    for chrom in PRIMARY_CHROMS:
        current = by_chrom.get(chrom)
        if current is None:
            rows.append(
                {
                    "chrom": chrom,
                    "chrom_length": int(chrom_lengths[chrom]),
                    "record_count": 0,
                    "raw_annotated_bp": 0,
                    "repeat_union_bp": 0,
                    "union_segment_count": 0,
                    "repeat_union_fraction": 0.0,
                }
            )
            continue
        starts = current["start0"].to_numpy()
        ends = current["end0"].to_numpy()
        covered_bp, union_segments = union_length(starts, ends)
        chrom_length = int(chrom_lengths[chrom])
        rows.append(
            {
                "chrom": chrom,
                "chrom_length": chrom_length,
                "record_count": current.height,
                "raw_annotated_bp": int((ends - starts).sum()),
                "repeat_union_bp": covered_bp,
                "union_segment_count": union_segments,
                "repeat_union_fraction": covered_bp / chrom_length,
            }
        )
    return pl.DataFrame(rows)


def build_category_inventory(annotations: pl.DataFrame) -> pl.DataFrame:
    length = (pl.col("end0") - pl.col("start0")).alias("length")
    base = annotations.with_columns(length)
    definitions = {
        "class": pl.col("repeat_class"),
        "family": pl.concat_str("repeat_class", "repeat_family", separator="|"),
        "subfamily": pl.concat_str(
            "repeat_class", "repeat_family", "repeat_name", separator="|"
        ),
    }
    summaries: list[pl.DataFrame] = []
    for level, expression in definitions.items():
        summaries.append(
            base.with_columns(expression.alias("label"))
            .group_by("label")
            .agg(
                pl.len().alias("record_count"),
                pl.col("length").sum().alias("raw_annotated_bp"),
                pl.col("chrom").n_unique().alias("chrom_count"),
                (pl.col("milli_div").mean() / 10.0).alias("mean_divergence_pct"),
            )
            .with_columns(pl.lit(level).alias("level"))
            .select(
                "level",
                "label",
                "record_count",
                "raw_annotated_bp",
                "chrom_count",
                "mean_divergence_pct",
            )
        )
    return pl.concat(summaries).sort(
        "level", "raw_annotated_bp", "label", descending=[False, True, False]
    )


def materialize(rmsk_path: Path, fai_path: Path, output_dir: Path) -> dict[str, Any]:
    assert not output_dir.exists()
    commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(commit)
    assert os.environ.get("RUN_ID") == RUN_ID
    chrom_lengths = read_fai(fai_path)
    annotations, source_rows = read_repeatmasker(rmsk_path, chrom_lengths)
    chrom_inventory = build_chrom_inventory(annotations, chrom_lengths)
    category_inventory = build_category_inventory(annotations)
    output_dir.mkdir(parents=True)
    annotation_path = output_dir / "annotations.parquet"
    chrom_path = output_dir / "chrom_inventory.parquet"
    category_path = output_dir / "category_inventory.parquet"
    annotations.write_parquet(annotation_path, compression="zstd")
    chrom_inventory.write_parquet(chrom_path, compression="zstd")
    category_inventory.write_parquet(category_path, compression="zstd")

    repeat_union_bp = int(chrom_inventory["repeat_union_bp"].sum())
    primary_genome_bp = sum(chrom_lengths.values())
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "experiment_commit": commit,
        "analysis_status": "outcome_blind_annotation_inventory",
        "coordinate_system": "0-based half-open",
        "source": {
            "url": RMSK_URL,
            "bytes": rmsk_path.stat().st_size,
            "sha256": sha256_file(rmsk_path),
            "source_rows": source_rows,
        },
        "fai": {
            "bytes": fai_path.stat().st_size,
            "sha256": sha256_file(fai_path),
        },
        "primary_chromosomes": list(PRIMARY_CHROMS),
        "primary_records": annotations.height,
        "filtered_nonprimary_records": source_rows - annotations.height,
        "primary_genome_bp": primary_genome_bp,
        "repeat_union_bp": repeat_union_bp,
        "repeat_union_fraction": repeat_union_bp / primary_genome_bp,
        "category_counts": {
            level: category_inventory.filter(pl.col("level") == level).height
            for level in ("class", "family", "subfamily")
        },
        "category_coverage_note": (
            "raw_annotated_bp sums source intervals and can double-count overlaps; "
            "repeat_union_bp is exact across all records. Primary focal labels are "
            "assigned only during panel construction."
        ),
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (annotation_path, chrom_path, category_path, result_path)
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmsk", type=Path, required=True)
    parser.add_argument("--fai", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.rmsk, args.fai, args.output_dir)


if __name__ == "__main__":
    main()
