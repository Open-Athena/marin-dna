"""Assembly of immutable #417 controls with new #473 enhancer outputs."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.issue_473.catalog import STANDARD_REGIONS
from marin_dna_vertebrate_projection.split import (
    add_stable_row_ids,
    assign_train_validation_splits,
)


def write_full_window_sequence_union(
    issue417_sequences_path: str | Path,
    enhancer_sequences_path: str | Path,
    output_path: str | Path,
) -> None:
    """Reuse four #417 regions and append only the new enhancer full-window arm."""
    baseline = pl.scan_parquet(issue417_sequences_path).filter(
        pl.col("region_label").is_in(STANDARD_REGIONS)
    )
    enhancer = pl.scan_parquet(enhancer_sequences_path)
    assert baseline.collect_schema() == enhancer.collect_schema()
    enhancer_regions = (
        enhancer.select("region_label")
        .unique()
        .collect(engine="streaming")["region_label"]
        .to_list()
    )
    assert enhancer_regions == ["ccre_enhancer_centered"]
    baseline_names = (
        baseline.select("query_name").unique().collect(engine="streaming")["query_name"]
    )
    enhancer_names = (
        enhancer.select("query_name").unique().collect(engine="streaming")["query_name"]
    )
    assert set(baseline_names).isdisjoint(enhancer_names)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.concat([baseline, enhancer], how="vertical").sink_parquet(output)


def write_full_window_qc_union(
    issue417_qc_path: str | Path,
    enhancer_qc_path: str | Path,
    output_path: str | Path,
) -> None:
    """Filter a #417 QC table to four regions and append matched enhancer QC."""
    baseline = pl.scan_parquet(issue417_qc_path).filter(
        pl.col("region_label").is_in(STANDARD_REGIONS)
    )
    enhancer = pl.scan_parquet(enhancer_qc_path)
    assert baseline.collect_schema() == enhancer.collect_schema()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.concat([baseline, enhancer], how="vertical").sink_parquet(output)


def write_baseline_compatibility_receipt(
    issue417_labels_path: str | Path,
    issue417_species_path: str | Path,
    active_species_path: str | Path,
    fixed_catalog_path: str | Path,
    output_path: str | Path,
) -> None:
    """Prove the reused baseline anchor and species identities remain compatible."""
    labels = (
        pl.read_parquet(issue417_labels_path)
        .filter(pl.col("label").is_in(STANDARD_REGIONS))
        .select(
            pl.col("name").alias("query_name"),
            (pl.lit("chr") + pl.col("chrom")).alias("source_chrom"),
            pl.col("start").alias("source_start"),
            pl.col("end").alias("source_end"),
            pl.col("label").alias("region_label"),
        )
        .sort("query_name")
    )
    fixed = (
        pl.read_parquet(fixed_catalog_path)
        .filter(pl.col("region_label").is_in(STANDARD_REGIONS))
        .select(labels.columns)
        .sort("query_name")
    )
    assert labels.equals(fixed), "fixed standard anchors differ from immutable #417"

    old_species = pl.read_csv(issue417_species_path, separator="\t")
    current_species = pl.read_csv(active_species_path, separator="\t")
    identity_columns = [
        "alignment_name",
        "scientific_name",
        "assembly",
        "taxonomy_id",
        "family",
        "clade",
        "backend",
    ]
    old_identity = old_species.select(identity_columns).sort("alignment_name")
    current_identity = current_species.select(identity_columns).sort("alignment_name")
    assert old_identity.equals(current_identity), (
        "active projection species differ from immutable #417"
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "standard_anchor_rows": labels.height,
                "active_target_species": old_identity.height,
                "anchor_identity": "exact dataframe equality",
                "species_identity": "exact dataframe equality",
                "standard_regions": list(STANDARD_REGIONS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_intersection_validation_views(
    full_window_path: str | Path,
    center_1_path: str | Path,
    full_window_output_path: str | Path,
    center_1_output_path: str | Path,
    selection_path: str | Path,
    summary_path: str | Path,
    *,
    region_label: str,
    validation_chrom: str = "chr18",
    max_validation_rows: int = 16_384,
    seed: int = 473,
) -> None:
    """Write matched validation rows selected from the accepted policy intersection."""
    full = (
        pl.scan_parquet(full_window_path)
        .filter(
            (pl.col("region_label") == region_label)
            & (pl.col("source_chrom") == validation_chrom)
        )
        .collect(engine="streaming")
    )
    center = (
        pl.scan_parquet(center_1_path)
        .filter(
            (pl.col("region_label") == region_label)
            & (pl.col("source_chrom") == validation_chrom)
        )
        .collect(engine="streaming")
    )
    key_columns = ["query_name", "species"]
    common_keys = full.select(key_columns).join(
        center.select(key_columns),
        on=key_columns,
        how="inner",
        validate="1:1",
    )
    assert common_keys.height > 0
    full_common = full.join(common_keys, on=key_columns, how="inner", validate="1:1")
    center_common = center.join(
        common_keys, on=key_columns, how="inner", validate="1:1"
    )
    full_with_ids = add_stable_row_ids(full_common)
    selected = assign_train_validation_splits(
        full_with_ids,
        validation_chrom=validation_chrom,
        max_validation_rows=max_validation_rows,
        seed=seed,
    )
    selected_ids = selected.validation.select("row_id")
    center_with_ids = add_stable_row_ids(center_common)
    center_validation = center_with_ids.join(
        selected_ids,
        on="row_id",
        how="inner",
        validate="1:1",
    )
    assert selected.validation.height == center_validation.height
    assert set(selected.validation["row_id"]) == set(center_validation["row_id"])

    full_output = Path(full_window_output_path)
    center_output = Path(center_1_output_path)
    selection_output = Path(selection_path)
    summary_output = Path(summary_path)
    for path in [full_output, center_output, selection_output, summary_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    selected.validation.sort("row_id").write_parquet(full_output)
    center_validation.sort("row_id").write_parquet(center_output)
    selected.selection_manifest.write_csv(selection_output, separator="\t")
    summary_output.write_text(
        json.dumps(
            {
                "region_label": region_label,
                "validation_chrom": validation_chrom,
                "seed": seed,
                "eligible_intersection_rows": common_keys.height,
                "selected_rows_per_policy": selected.validation.height,
                "realized_token_count_per_policy": selected.realized_token_count,
                "row_identity": "query_name, species, source interval, backend",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
