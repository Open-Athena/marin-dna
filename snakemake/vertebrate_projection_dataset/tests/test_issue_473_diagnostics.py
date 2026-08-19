from __future__ import annotations

from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.issue_473.diagnostics import (
    build_paired_union,
    write_paired_diagnostics,
)


def _accepted(
    query_name: str,
    *,
    species: str = "Mus musculus",
    t_chrom: str = "chr5",
    t_start: int = 100,
    strand: str = "+",
    aligned_bases: int = 255,
    sequence: str = "A" * 255,
) -> dict[str, object]:
    return {
        "query_name": query_name,
        "species": species,
        "source_chrom": "chr1",
        "source_start": 0,
        "source_end": 255,
        "region_label": "cds",
        "alignment_source": "zoonomia_cactus",
        "assembly": "mm10",
        "taxonomy_id": 10090,
        "family": "Muridae",
        "clade": "mammals",
        "phylogenetic_rank": 1,
        "t_chrom": t_chrom,
        "t_start": t_start,
        "t_end": t_start + 255,
        "t_strand": strand,
        "pre_resize_t_start": t_start + 100,
        "pre_resize_t_end": t_start + 101,
        "fragment_count": 1,
        "aligned_bases": aligned_bases,
        "sequence": sequence,
    }


def _rejections(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "query_name": query_name,
                "species": "Mus musculus",
                "rejection_reason": reason,
                "detail": "test",
            }
            for query_name, reason in rows
        ],
        schema={
            "query_name": pl.String,
            "species": pl.String,
            "rejection_reason": pl.String,
            "detail": pl.String,
        },
    )


def test_paired_union_reports_status_locus_flanks_and_sequence_metrics() -> None:
    full = pl.DataFrame(
        [
            _accepted("both", sequence="g" * 255),
            _accepted("full-only"),
        ]
    )
    center = pl.DataFrame(
        [
            _accepted("both", t_start=110, aligned_bases=1, sequence="N" * 255),
            _accepted("center-only", aligned_bases=1),
        ]
    )
    result = build_paired_union(
        full.lazy(),
        center.lazy(),
        _rejections([("center-only", "span_too_long")]).lazy(),
        _rejections([("full-only", "no_unique_locus")]).lazy(),
    ).collect()

    both = result.filter(pl.col("query_name") == "both").row(0, named=True)
    assert both["pair_outcome"] == "both"
    assert both["target_chrom_agreement"]
    assert both["target_strand_agreement"]
    assert both["target_locus_overlap"]
    assert not both["target_locus_exact"]
    assert both["emitted_center_displacement_bases"] == 10
    assert both["full_window_landmark_aligned_fraction"] == 1.0
    assert both["center_1_landmark_aligned_fraction"] == 1.0
    assert both["center_1_human_oriented_left_flank_bases"] == 100
    assert both["center_1_human_oriented_right_flank_bases"] == 154
    assert both["full_window_gc_fraction"] == 1.0
    assert both["center_1_ambiguous_base_fraction"] == 1.0
    assert both["emitted_window_aligned_coverage"] is None
    assert both["emitted_window_aligned_coverage_status"] == "unavailable_genome_wide"

    full_only = result.filter(pl.col("query_name") == "full-only").row(0, named=True)
    assert full_only["center_1_status"] == "rejected:no_unique_locus"
    center_only = result.filter(pl.col("query_name") == "center-only").row(
        0, named=True
    )
    assert center_only["full_window_status"] == "rejected:span_too_long"


def test_writer_clusters_recovery_uncertainty_by_anchor(tmp_path: Path) -> None:
    full = pl.DataFrame([_accepted("a1"), _accepted("a2")])
    center = pl.DataFrame(
        [
            _accepted("a1", aligned_bases=1),
            _accepted("a1", species="Bos taurus", aligned_bases=1),
        ]
    )
    full_path = tmp_path / "full.parquet"
    center_path = tmp_path / "center.parquet"
    full_rejected_path = tmp_path / "full-rejected.parquet"
    center_rejected_path = tmp_path / "center-rejected.parquet"
    anchors_path = tmp_path / "anchors.parquet"
    full.write_parquet(full_path)
    center.write_parquet(center_path)
    _rejections([]).write_parquet(full_rejected_path)
    _rejections([]).write_parquet(center_rejected_path)
    pl.DataFrame(
        {
            "query_name": ["a1", "a2", "a3"],
            "source_chrom": ["chr1"] * 3,
            "source_start": [0, 255, 510],
            "source_end": [255, 510, 765],
            "region_label": ["cds"] * 3,
        }
    ).write_parquet(anchors_path)

    paired = tmp_path / "paired.parquet"
    scopes = tmp_path / "scopes.parquet"
    per_anchor = tmp_path / "per-anchor.parquet"
    uncertainty = tmp_path / "uncertainty.parquet"
    write_paired_diagnostics(
        full_path,
        center_path,
        [str(full_rejected_path)],
        [str(center_rejected_path)],
        anchors_path,
        paired,
        scopes,
        per_anchor,
        uncertainty,
    )

    anchors = pl.read_parquet(per_anchor).sort("query_name")
    assert anchors["accepted_species_delta"].to_list() == [1, -1, 0]
    summary = pl.read_parquet(uncertainty).row(0, named=True)
    assert summary["n_anchors"] == 3
    assert summary["mean_paired_delta"] == 0.0
    assert summary["uncertainty_method"] == "anchor-clustered normal interval"
