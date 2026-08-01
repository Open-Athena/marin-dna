"""Validate the completed issue #417 full projection outputs.

This is intentionally a one-off, assertion-heavy audit over the final artifact
set.  It uses lazy scans and per-species sequence files so validation does not
materialize the combined projection dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import polars as pl


COHORTS = [
    "all",
    "cds",
    "cds_mammals_only",
    "utr3",
    "ncrna_exon",
    "tss_region_and_utr5",
    "ccre_non_promoter",
    "background",
]
TARGET_LENGTH = 255
VALIDATION_CHROM = "chr18"
VALIDATION_MAX_ROWS = 16_384
TARGET_BACKENDS = {"zoonomia_cactus": 107, "ucsc_multiz100way": 28}


def _single_row(scan: pl.LazyFrame, *expressions: pl.Expr) -> dict[str, Any]:
    return scan.select(*expressions).collect(engine="streaming").row(0, named=True)


def _assert_sequence_frame(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing sequence parquet: {path}"
    scan = pl.scan_parquet(path)
    required = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "species",
        "assembly",
        "taxonomy_id",
        "clade",
        "alignment_source",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
        "sequence",
    }
    missing = required - set(scan.collect_schema().names())
    assert not missing, f"{path} missing columns: {sorted(missing)}"
    critical = sorted(required)
    invalid = _single_row(
        scan,
        pl.len().alias("rows"),
        pl.col("query_name").n_unique().alias("queries"),
        pl.col("species").n_unique().alias("species_count"),
        pl.col("species").first().alias("species"),
        pl.col("alignment_source").first().alias("backend"),
        pl.any_horizontal([pl.col(column).is_null() for column in critical])
        .sum()
        .alias("null_rows"),
        (pl.col("source_start") < 0).sum().alias("negative_source_start"),
        (pl.col("source_end") - pl.col("source_start") != TARGET_LENGTH)
        .sum()
        .alias("invalid_source_span"),
        (pl.col("t_start") < 0).sum().alias("negative_target_start"),
        (pl.col("t_end") - pl.col("t_start") != TARGET_LENGTH)
        .sum()
        .alias("invalid_target_span"),
        (pl.col("t_end") > pl.col("t_src_size")).sum().alias("target_out_of_bounds"),
        (~pl.col("t_strand").is_in(["+", "-"])).sum().alias("invalid_strand"),
        (pl.col("sequence").str.len_bytes() != TARGET_LENGTH)
        .sum()
        .alias("invalid_sequence_length"),
    )
    rows = int(invalid["rows"])
    assert rows > 0, f"empty sequence parquet: {path}"
    assert rows == int(invalid["queries"]), f"duplicate query in {path}"
    assert int(invalid["species_count"]) == 1, f"multiple species in {path}"
    for key in [
        "null_rows",
        "negative_source_start",
        "invalid_source_span",
        "negative_target_start",
        "invalid_target_span",
        "target_out_of_bounds",
        "invalid_strand",
        "invalid_sequence_length",
    ]:
        assert int(invalid[key]) == 0, f"{path}: {key}={invalid[key]}"
    return {
        "rows": rows,
        "species": str(invalid["species"]),
        "backend": str(invalid["backend"]),
    }


def _validate_sequences(results: Path, manifest: pl.DataFrame) -> dict[str, object]:
    selected = manifest.filter(pl.col("selected"))
    assert selected.height == sum(TARGET_BACKENDS.values())
    observed_backend_counts = {
        str(backend): int(count)
        for backend, count in selected.group_by("backend").len().iter_rows()
    }
    assert observed_backend_counts == TARGET_BACKENDS
    assert selected["scientific_name"].n_unique() == selected.height
    assert selected["family"].n_unique() == selected.height

    file_stats: list[dict[str, Any]] = []
    human_stats = _assert_sequence_frame(results / "sequences/human_reference.parquet")
    assert human_stats["species"] == "Homo sapiens"
    assert human_stats["backend"] == "human_reference"
    file_stats.append(human_stats)

    for row in selected.iter_rows(named=True):
        backend = str(row["backend"])
        directory = "hal" if backend == "zoonomia_cactus" else "multiz"
        stats = _assert_sequence_frame(
            results / "sequences" / directory / f"{row['alignment_name']}.parquet"
        )
        assert stats["species"] == row["scientific_name"]
        assert stats["backend"] == backend
        file_stats.append(stats)

    combined_path = results / "sequences/all_sources.parquet"
    assert combined_path.is_file()
    combined = pl.scan_parquet(combined_path)
    combined_stats = _single_row(
        combined,
        pl.len().alias("rows"),
        pl.col("species").n_unique().alias("species"),
        pl.col("query_name")
        .str.to_lowercase()
        .str.starts_with("zrs_")
        .sum()
        .alias("zrs_rows"),
        (pl.col("sequence").str.len_bytes() != TARGET_LENGTH)
        .sum()
        .alias("invalid_sequence_length"),
        (pl.col("t_end") - pl.col("t_start") != TARGET_LENGTH)
        .sum()
        .alias("invalid_target_span"),
    )
    expected_rows = sum(int(stats["rows"]) for stats in file_stats)
    assert int(combined_stats["rows"]) == expected_rows
    assert int(combined_stats["species"]) == len(file_stats)
    assert int(combined_stats["zrs_rows"]) == 0, (
        "ZRS controls must remain outside the conservation-filtered full dataset"
    )
    assert int(combined_stats["invalid_sequence_length"]) == 0
    assert int(combined_stats["invalid_target_span"]) == 0

    observed_metadata = (
        combined.select(
            "species", "assembly", "taxonomy_id", "clade", "alignment_source"
        )
        .unique()
        .collect(engine="streaming")
    )
    assert observed_metadata.height == len(file_stats)
    target_metadata = observed_metadata.filter(
        pl.col("alignment_source") != "human_reference"
    )
    expected_metadata = selected.select(
        pl.col("scientific_name").alias("species"),
        "assembly",
        "taxonomy_id",
        "clade",
        pl.col("backend").alias("alignment_source"),
    )
    assert target_metadata.sort("species").equals(expected_metadata.sort("species"))
    assert target_metadata.filter(
        (pl.col("alignment_source") == "ucsc_multiz100way")
        & (pl.col("clade") == "mammals")
    ).is_empty()

    return {
        "rows": expected_rows,
        "species": len(file_stats),
        "human_rows": int(human_stats["rows"]),
        "backend_species": {"human_reference": 1, **observed_backend_counts},
    }


def _dataset_stats(path: Path, *, validation: bool) -> dict[str, object]:
    assert path.is_file(), f"missing dataset parquet: {path}"
    scan = pl.scan_parquet(path)
    stats = _single_row(
        scan,
        pl.len().alias("rows"),
        (pl.col("sequence").str.len_bytes() != TARGET_LENGTH)
        .sum()
        .alias("invalid_sequence_length"),
        (
            pl.col("source_chrom") == VALIDATION_CHROM
            if not validation
            else pl.col("source_chrom") != VALIDATION_CHROM
        )
        .sum()
        .alias("invalid_chrom"),
        (~pl.col("augmentation").is_in(["+"] if validation else ["+", "-"]))
        .sum()
        .alias("invalid_augmentation"),
        pl.col("query_name")
        .str.to_lowercase()
        .str.starts_with("zrs_")
        .sum()
        .alias("zrs_rows"),
        pl.col("species").n_unique().alias("species"),
    )
    for key in [
        "invalid_sequence_length",
        "invalid_chrom",
        "invalid_augmentation",
        "zrs_rows",
    ]:
        assert int(stats[key]) == 0, f"{path}: {key}={stats[key]}"
    if validation:
        assert int(stats["rows"]) <= VALIDATION_MAX_ROWS
    else:
        augmentation_counts = {
            str(augmentation): int(count)
            for augmentation, count in (
                scan.group_by("augmentation")
                .len()
                .collect(engine="streaming")
                .iter_rows()
            )
        }
        assert set(augmentation_counts) == {"+", "-"}
        assert augmentation_counts["+"] == augmentation_counts["-"]
    return {"rows": int(stats["rows"]), "species": int(stats["species"])}


def _frame_digest(scan: pl.LazyFrame) -> dict[str, int]:
    columns = scan.collect_schema().names()
    hashed = scan.select(pl.struct(columns).hash(seed=417).alias("row_hash"))
    stats = _single_row(
        hashed,
        pl.len().alias("rows"),
        pl.col("row_hash").sum().alias("hash_sum"),
        pl.col("row_hash").min().alias("hash_min"),
        pl.col("row_hash").max().alias("hash_max"),
    )
    return {key: int(stats[key]) for key in stats}


def _validate_datasets(
    results: Path, expected_pipeline_commit: str | None
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for cohort in COHORTS:
        directory = results / "datasets" / cohort
        train_path = directory / "train.parquet"
        validation_path = directory / "validation.parquet"
        train = _dataset_stats(train_path, validation=False)
        validation = _dataset_stats(validation_path, validation=True)

        summary = json.loads((directory / "split_summary.json").read_text())
        assert summary["region_label"] == (
            "cds" if cohort == "cds_mammals_only" else cohort
        )
        assert summary["species_scope"] == (
            "mammals_only" if cohort == "cds_mammals_only" else "all"
        )
        assert summary["validation_chrom"] == VALIDATION_CHROM
        assert int(summary["train_rows"]) == train["rows"]
        assert int(summary["validation_rows"]) == validation["rows"]
        assert int(summary["realized_token_count"]) == validation["rows"] * 256

        selection = pl.read_csv(directory / "validation_selection.tsv", separator="\t")
        species_counts = pl.read_csv(
            directory / "validation_species_counts.tsv", separator="\t"
        )
        assert selection.height == validation["rows"]
        assert int(species_counts["selected_rows"].sum()) == validation["rows"]
        assert species_counts["species"].n_unique() == species_counts.height
        assert species_counts.height == validation["species"]
        if int(summary["eligible_validation_rows"]) >= VALIDATION_MAX_ROWS:
            assert validation["rows"] == VALIDATION_MAX_ROWS

        card = (directory / "README.md").read_text()
        for tag in ["biology", "genomics", "dna"]:
            assert re.search(rf"(?m)^- {tag}$", card), f"{cohort} card missing {tag}"
        if expected_pipeline_commit is not None:
            assert f"blob/{expected_pipeline_commit}/" in card
        assert f"`train`: {train['rows']:,} rows" in card
        assert f"`validation`: {validation['rows']:,}" in card
        summaries[cohort] = {"train": train, "validation": validation}

    mammals_train = pl.scan_parquet(results / "datasets/cds_mammals_only/train.parquet")
    cds_mammal_subset = pl.scan_parquet(results / "datasets/cds/train.parquet").filter(
        pl.col("alignment_source") != "ucsc_multiz100way"
    )
    assert _frame_digest(mammals_train) == _frame_digest(cds_mammal_subset), (
        "CDS mammals-only train is not exactly CDS train with MultiZ rows removed"
    )
    mammals_sources = (
        mammals_train.select("alignment_source")
        .unique()
        .collect(engine="streaming")["alignment_source"]
        .to_list()
    )
    assert set(mammals_sources) == {"human_reference", "zoonomia_cactus"}
    mammals_validation_sources = (
        pl.scan_parquet(results / "datasets/cds_mammals_only/validation.parquet")
        .select("alignment_source")
        .unique()
        .collect(engine="streaming")["alignment_source"]
        .to_list()
    )
    assert set(mammals_validation_sources) == {
        "human_reference",
        "zoonomia_cactus",
    }
    return summaries


def _validate_qc(results: Path, target_species: int) -> dict[str, object]:
    anchors = pl.scan_parquet(results / "anchors/catalog.parquet")
    per_anchor = pl.scan_parquet(results / "qc/per_anchor.parquet")
    anchor_rows = int(_single_row(anchors, pl.len().alias("rows"))["rows"])
    stats = _single_row(
        per_anchor,
        pl.len().alias("rows"),
        pl.col("query_name").n_unique().alias("queries"),
        (pl.col("requested_total_species") != target_species)
        .sum()
        .alias("invalid_requested"),
        (
            pl.col("accepted_mammal_projections")
            + pl.col("accepted_non_mammal_projections")
            != pl.col("accepted_total_projections")
        )
        .sum()
        .alias("invalid_accepted"),
        (
            (pl.col("recovered_fraction") < 0)
            | (pl.col("recovered_fraction") > 1)
            | (pl.col("no_mapping_count") < 0)
        )
        .sum()
        .alias("invalid_ranges"),
    )
    assert int(stats["rows"]) == anchor_rows == int(stats["queries"])
    assert int(stats["invalid_requested"]) == 0
    assert int(stats["invalid_accepted"]) == 0
    assert int(stats["invalid_ranges"]) == 0

    rejection_total = int(
        _single_row(
            pl.scan_parquet(results / "qc/rejection_counts.parquet"),
            pl.col("count").sum().alias("count"),
        )["count"]
    )
    accepted_total = int(
        _single_row(
            per_anchor,
            pl.col("accepted_total_projections").sum().alias("count"),
        )["count"]
    )
    assert accepted_total + rejection_total == anchor_rows * target_species

    breadth = (
        per_anchor.group_by("region_label")
        .agg(
            pl.len().alias("anchors"),
            pl.col("accepted_mammal_projections").mean().alias("mean_mammals"),
            pl.col("accepted_non_mammal_projections").mean().alias("mean_non_mammals"),
            pl.col("accepted_total_projections").mean().alias("mean_total"),
        )
        .sort("region_label")
        .collect(engine="streaming")
    )
    breadth_by_region = {
        str(row["region_label"]): {
            "anchors": int(row["anchors"]),
            "mean_mammals": float(row["mean_mammals"]),
            "mean_non_mammals": float(row["mean_non_mammals"]),
            "mean_total": float(row["mean_total"]),
        }
        for row in breadth.to_dicts()
    }
    assert "cds" in breadth_by_region and "ccre_non_promoter" in breadth_by_region

    report = (results / "qc/manual_inspection.md").read_text().lower()
    assert "pending human review" in report
    assert "zrs positive control: separate sidecar qc" in report
    assert "intentionally absent from the conservation-filtered grid" in report
    return {
        "anchors": anchor_rows,
        "accepted_target_projections": accepted_total,
        "rejected_or_unmapped_target_projections": rejection_total,
        "breadth_by_region": breadth_by_region,
        "cds_broader_than_ccre": (
            breadth_by_region["cds"]["mean_total"]
            > breadth_by_region["ccre_non_promoter"]["mean_total"]
        ),
        "zrs_rows": 0,
        "zrs_qc": "validated separately as a sidecar",
    }


def validate_outputs(
    results: Path,
    *,
    expected_pipeline_commit: str | None,
) -> dict[str, object]:
    assert results.is_dir(), f"missing results directory: {results}"
    if expected_pipeline_commit is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", expected_pipeline_commit)
    manifest = pl.read_csv(results / "metadata/species_active.tsv", separator="\t")
    assert manifest["selected"].all()
    sequences = _validate_sequences(results, manifest)
    datasets = _validate_datasets(results, expected_pipeline_commit)
    qc = _validate_qc(results, sum(TARGET_BACKENDS.values()))
    return {
        "results": str(results.resolve()),
        "expected_pipeline_commit": expected_pipeline_commit,
        "sequences": sequences,
        "datasets": datasets,
        "qc": qc,
        "manual_inspection": "pending human review",
        "status": "automated validation passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--expected-pipeline-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = validate_outputs(
        args.results,
        expected_pipeline_commit=args.expected_pipeline_commit,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
