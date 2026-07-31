"""Thin, assertion-heavy file wrappers used by the Snakemake pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import py2bit

from marin_dna.pipelines.projection.dataset import reverse_complement_col
from marin_dna.pipelines.vertebrate_projection_dataset.adapters import (
    build_human_reference_rows,
    hal_records_to_fragments,
)
from marin_dna.pipelines.vertebrate_projection_dataset.contract import (
    apply_projection_contract,
    extract_oriented_sequences,
)
from marin_dna.pipelines.vertebrate_projection_dataset.inspection import (
    assert_zrs_broad_recovery,
    build_inspection_sample,
    build_rejection_inspection_sample,
    render_inspection_report,
)
from marin_dna.pipelines.vertebrate_projection_dataset.maf import (
    project_anchors_from_maf,
)
from marin_dna.pipelines.vertebrate_projection_dataset.manifest import (
    read_species_manifest,
)
from marin_dna.pipelines.vertebrate_projection_dataset.qc import (
    build_projection_qc_tables,
)
from marin_dna.pipelines.vertebrate_projection_dataset.split import (
    assign_train_validation_splits,
)


def read_anchor_catalog(path: str | Path, *, target_length: int = 255) -> pl.DataFrame:
    """Read a TSV/Parquet anchor catalog and assert 0-based half-open invariants."""
    anchor_path = Path(path)
    frame = (
        pl.read_parquet(anchor_path)
        if anchor_path.suffix == ".parquet"
        else pl.read_csv(anchor_path, separator="\t")
    )
    required = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
    }
    missing = required - set(frame.columns)
    assert not missing, f"anchor catalog missing columns: {sorted(missing)}"
    assert frame["query_name"].n_unique() == frame.height
    assert frame["source_chrom"].str.starts_with("chr").all()
    assert (frame["source_start"] >= 0).all()
    assert (frame["source_end"] - frame["source_start"] == target_length).all()
    return frame.select(sorted(required)).sort(
        "source_chrom", "source_start", "query_name"
    )


def write_hal_bed6(anchors_path: str | Path, output_path: str | Path) -> None:
    anchors = read_anchor_catalog(anchors_path)
    anchors.select(
        pl.col("source_chrom"),
        pl.col("source_start"),
        pl.col("source_end"),
        pl.col("query_name"),
        pl.lit(0).alias("score"),
        pl.lit("+").alias("strand"),
    ).write_csv(output_path, separator="\t", include_header=False)


def write_maf_candidates(
    maf_path: str | Path,
    anchors_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> None:
    anchors = read_anchor_catalog(anchors_path)
    manifest = read_species_manifest(str(manifest_path))
    project_anchors_from_maf(maf_path, anchors, manifest).write_parquet(output_path)


def write_hal_fragments(
    hal_records: pl.DataFrame,
    anchors_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> None:
    anchors = read_anchor_catalog(anchors_path)
    manifest = read_species_manifest(str(manifest_path))
    hal_records_to_fragments(hal_records, anchors, manifest).write_parquet(output_path)


def write_contract_outputs(
    fragments_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    *,
    target_length: int,
    pre_resize_min_length: int,
    pre_resize_max_length: int,
) -> None:
    result = apply_projection_contract(
        pl.read_parquet(fragments_path),
        target_length=target_length,
        pre_resize_min_length=pre_resize_min_length,
        pre_resize_max_length=pre_resize_max_length,
    )
    result.accepted.write_parquet(accepted_path)
    result.rejected.write_parquet(rejected_path)


def write_twobit_sequences(
    accepted_path: str | Path,
    two_bit_path: str | Path,
    sequence_path: str | Path,
    rejected_path: str | Path,
    *,
    target_length: int = 255,
) -> None:
    accepted = pl.read_parquet(accepted_path)
    genome = py2bit.open(str(two_bit_path))

    def fetch(_assembly: str, chrom: str, start: int, end: int) -> str | None:
        return genome.sequence(chrom, start, end)

    try:
        result = extract_oriented_sequences(
            accepted, fetch, target_length=target_length
        )
    finally:
        genome.close()
    result.accepted.write_parquet(sequence_path)
    result.rejected.write_parquet(rejected_path)


def write_human_reference_sequences(
    anchors_path: str | Path,
    two_bit_path: str | Path,
    chrom_sizes_path: str | Path,
    output_path: str | Path,
    *,
    target_length: int = 255,
) -> None:
    anchors = read_anchor_catalog(anchors_path, target_length=target_length)
    sizes = pl.read_csv(
        chrom_sizes_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "size"],
    )
    chromosome_sizes = dict(sizes.iter_rows())
    genome = py2bit.open(str(two_bit_path))

    def fetch(_assembly: str, chrom: str, start: int, end: int) -> str | None:
        return genome.sequence(chrom, start, end)

    try:
        rows = build_human_reference_rows(
            anchors,
            chromosome_sizes,
            fetch,
            target_length=target_length,
        )
    finally:
        genome.close()
    rows.write_parquet(output_path)


def combine_sequence_parquets(input_paths: list[str], output_path: str | Path) -> None:
    assert input_paths
    frames = [pl.read_parquet(path) for path in input_paths]
    columns = frames[0].columns
    assert all(frame.columns == columns for frame in frames)
    combined = pl.concat(frames)
    assert combined.select("query_name", "species").is_unique().all()
    assert (combined["sequence"].str.len_bytes() == 255).all()
    combined.sort("query_name", "species").write_parquet(output_path)


def write_dataset_split_files(
    combined_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    selection_path: str | Path,
    species_counts_path: str | Path,
    summary_path: str | Path,
    *,
    region_label: str,
    add_rc: bool,
    validation_chrom: str,
    max_validation_rows: int,
    seed: int,
) -> None:
    original = pl.read_parquet(combined_path)
    if region_label != "all":
        original = original.filter(pl.col("region_label") == region_label)
    assert not original.is_empty(), f"empty region cohort: {region_label}"
    if add_rc:
        rows = pl.concat(
            [
                original.with_columns(pl.lit("+").alias("augmentation")),
                original.with_columns(
                    reverse_complement_col(pl.col("sequence")).alias("sequence"),
                    pl.lit("-").alias("augmentation"),
                ),
            ]
        )
    else:
        rows = original.with_columns(pl.lit("+").alias("augmentation"))
    result = assign_train_validation_splits(
        rows,
        validation_chrom=validation_chrom,
        max_validation_rows=max_validation_rows,
        seed=seed,
    )
    result.train.write_parquet(train_path)
    result.validation.write_parquet(validation_path)
    result.selection_manifest.write_csv(selection_path, separator="\t")
    result.species_counts.write_csv(species_counts_path, separator="\t")
    Path(summary_path).write_text(
        json.dumps(
            {
                "region_label": region_label,
                "seed": seed,
                "validation_chrom": validation_chrom,
                "validation_rows": result.validation.height,
                "realized_token_count": result.realized_token_count,
            },
            indent=2,
        )
        + "\n"
    )


def write_qc_files(
    anchors_path: str | Path,
    accepted_path: str | Path,
    rejected_paths: list[str],
    manifest_path: str | Path,
    per_anchor_path: str | Path,
    per_scope_path: str | Path,
    rejections_path: str | Path,
    aggregates_path: str | Path,
    *,
    validation_chrom: str,
) -> None:
    anchors = read_anchor_catalog(anchors_path).with_columns(
        pl.when(pl.col("source_chrom") == validation_chrom)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )
    rejected_frames = [pl.read_parquet(path) for path in rejected_paths]
    rejected = pl.concat(rejected_frames) if rejected_frames else pl.DataFrame()
    tables = build_projection_qc_tables(
        anchors,
        pl.read_parquet(accepted_path),
        rejected,
        read_species_manifest(str(manifest_path)),
    )
    tables.per_anchor.write_parquet(per_anchor_path)
    tables.per_anchor_scope.write_parquet(per_scope_path)
    tables.rejection_counts.write_parquet(rejections_path)
    tables.aggregates.write_parquet(aggregates_path)


def write_inspection_files(
    sequences_path: str | Path,
    rejected_paths: list[str],
    sample_path: str | Path,
    rejected_sample_path: str | Path,
    report_path: str | Path,
    *,
    seed: int,
    rows_per_region: int,
    fragmented_rows: int,
    rejected_rows_per_reason: int,
) -> None:
    """Write a deterministic sample plus an explicitly pending review report."""
    sequences = pl.read_parquet(sequences_path)
    assert_zrs_broad_recovery(sequences)
    sample = build_inspection_sample(
        sequences,
        seed=seed,
        rows_per_region=rows_per_region,
        fragmented_rows=fragmented_rows,
    )
    rejected = pl.concat([pl.read_parquet(path) for path in rejected_paths])
    rejected_sample = build_rejection_inspection_sample(
        rejected, seed=seed, rows_per_reason=rejected_rows_per_reason
    )
    sample.write_csv(sample_path, separator="\t")
    rejected_sample.write_csv(rejected_sample_path, separator="\t")
    Path(report_path).write_text(
        render_inspection_report(sample, rejected_sample, seed=seed)
    )


def write_dataset_card(
    train_path: str | Path,
    validation_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    pipeline_commit: str,
    hf_repo: str,
    region_label: str,
) -> None:
    """Write the reviewable HF README required before any upload."""
    assert len(pipeline_commit) == 40, "dataset cards require a commit-pinned SHA"
    train = pl.read_parquet(train_path)
    validation = pl.read_parquet(validation_path)
    selected = read_species_manifest(str(manifest_path)).filter(pl.col("selected"))
    species_counts = (
        selected.group_by("backend", "clade")
        .len(name="species")
        .sort("backend", "clade")
    )
    species_lines = "\n".join(
        f"| {backend} | {clade} | {count} |"
        for backend, clade, count in species_counts.iter_rows()
    )
    schema_lines = "\n".join(
        f"- `{column}`: `{dtype}`" for column, dtype in train.schema.items()
    )
    pipeline_url = (
        "https://github.com/Open-Athena/marin-dna/blob/"
        f"{pipeline_commit}/snakemake/vertebrate_projection_dataset/README.md"
    )
    text = f"""---
tags:
- biology
- genomics
- dna
---

# `{hf_repo}`

Human-anchored 255 bp vertebrate sequences from the Zoonomia 447-mammal
Cactus alignment and UCSC hg38 MultiZ 100-way alignment. This draft covers
the `{region_label}` region cohort and preserves source FASTA/2bit letter case.

Produced by the [commit-pinned vertebrate projection pipeline]({pipeline_url}).

## Splits

- `train`: {train.height:,} rows; no chromosome-18 source anchors.
- `validation`: {validation.height:,} original-orientation chromosome-18 rows
  ({validation.height * 256:,} tokens including BOS).

The selected target manifest contains {selected.height:,} family-deduplicated
projection targets; human reference rows are added separately once per anchor.

| Projection backend | Clade | Selected species |
|---|---|---:|
{species_lines}

## Schema

{schema_lines}
"""
    Path(output_path).write_text(text)
