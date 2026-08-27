from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.phylop_uniform import (
    PHYLOP_ASSIGNMENT_RECIPE,
    PHYLOP_UNIFORM_ARMS,
    write_phylop_uniform_anchor_catalog,
)


def _labels() -> pl.DataFrame:
    labels = [
        "cds",
        "utr3",
        "tss_region_and_utr5",
        "ncrna_exon",
        "ccre_non_promoter",
        "ccre_non_promoter",
        "background",
    ]
    return pl.DataFrame(
        {
            "name": [f"w{index}" for index in range(len(labels))],
            "chrom": ["1"] * len(labels),
            "start": [index * 128 for index in range(len(labels))],
            "end": [index * 128 + 255 for index in range(len(labels))],
            "label": labels,
            "functional_frac": [1.0] * 6 + [0.0],
            "cds_frac": [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            "utr3_frac": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "tss_region_and_utr5_frac": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "ncrna_exon_frac": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "ccre_non_promoter_frac": [0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0],
            "gene_body_frac": [0.0] * len(labels),
            "intron_frac": [0.0] * len(labels),
            "intergenic_frac": [1.0] * len(labels),
        }
    )


def test_write_phylop_uniform_catalog_is_exhaustive_and_pinned(
    tmp_path: Path,
) -> None:
    labels = _labels()
    labels_path = tmp_path / "labels.parquet"
    scored_path = tmp_path / "scored.parquet"
    catalog_path = tmp_path / "catalog.parquet"
    assignments_path = tmp_path / "assignments.parquet"
    summary_path = tmp_path / "summary.json"
    labels.write_parquet(labels_path)
    labels.select("name", "start", "end").with_columns(
        pl.lit("chr1").alias("chrom"),
        pl.lit(51).cast(pl.Int32).alias("conserved_bases"),
        pl.lit(0.2).cast(pl.Float32).alias("proportion_conserved"),
        pl.lit(2.5).cast(pl.Float32).alias("mean_phylop"),
        pl.lit(255).cast(pl.Int32).alias("n_valid_bases"),
    ).write_parquet(scored_path)

    observed = write_phylop_uniform_anchor_catalog(
        labels_path,
        [scored_path],
        catalog_path,
        assignments_path,
        summary_path,
        phylop_track="phyloP_447m",
        phylop_threshold=2.2162,
        min_proportion_conserved=0.20,
        expected_full_count=labels.height,
    )

    catalog = pl.read_parquet(catalog_path)
    assignments = pl.read_parquet(assignments_path)
    assert catalog.height == assignments.height == labels.height
    assert set(catalog["region_label"].to_list()) == set(PHYLOP_UNIFORM_ARMS)
    assert assignments["assignment_recipe"].unique().to_list() == [
        PHYLOP_ASSIGNMENT_RECIPE
    ]
    assert assignments["phylop_track"].unique().to_list() == ["phyloP_447m"]
    assert observed["assignment_arm_count_sum"] == labels.height
    assert json.loads(summary_path.read_text()) == observed


def test_phylop_smoke_catalog_selects_each_arm(tmp_path: Path) -> None:
    labels = _labels()
    labels_path = tmp_path / "labels.parquet"
    scored_path = tmp_path / "scored.parquet"
    labels.write_parquet(labels_path)
    labels.select("name", "start", "end").with_columns(
        pl.lit("chr1").alias("chrom"),
        pl.lit(51).cast(pl.Int32).alias("conserved_bases"),
        pl.lit(0.2).cast(pl.Float32).alias("proportion_conserved"),
        pl.lit(2.5).cast(pl.Float32).alias("mean_phylop"),
        pl.lit(255).cast(pl.Int32).alias("n_valid_bases"),
    ).write_parquet(scored_path)

    summary = write_phylop_uniform_anchor_catalog(
        labels_path,
        [scored_path],
        tmp_path / "catalog.parquet",
        tmp_path / "assignments.parquet",
        tmp_path / "summary.json",
        phylop_track="phyloP_447m",
        phylop_threshold=2.2162,
        min_proportion_conserved=0.20,
        smoke_anchors_per_arm=1,
        allowed_chroms=["chr1"],
    )

    assert summary["catalog_rows"] == len(PHYLOP_UNIFORM_ARMS)
    assert summary["pre_smoke_cap_rows"] == labels.height
    assert summary["allowed_chroms"] == ["chr1"]
