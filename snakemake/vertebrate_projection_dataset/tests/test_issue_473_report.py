from __future__ import annotations

import polars as pl
from marin_dna_vertebrate_projection.issue_473.report import (
    build_outcome_counts,
    summarize_accepted_rows,
)


def _accepted(
    query_name: str,
    species: str,
    *,
    region: str = "cds",
    aligned_bases: int = 255,
    fragment_count: int = 1,
) -> dict[str, object]:
    return {
        "query_name": query_name,
        "species": species,
        "region_label": region,
        "alignment_source": "zoonomia_cactus",
        "clade": "mammals",
        "t_start": 100,
        "t_end": 355,
        "t_strand": "+",
        "pre_resize_t_start": 100,
        "pre_resize_t_end": 355,
        "fragment_count": fragment_count,
        "aligned_bases": aligned_bases,
        "sequence": "acGTN" * 51,
    }


def _rejected(rows: list[tuple[str, str, str]]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "query_name": [row[0] for row in rows],
            "species": [row[1] for row in rows],
            "rejection_reason": [row[2] for row in rows],
        },
        schema={
            "query_name": pl.String,
            "species": pl.String,
            "rejection_reason": pl.String,
        },
    ).lazy()


def test_summarize_accepted_rows_reports_geometry_and_sequence_contracts() -> None:
    rows = pl.DataFrame(
        [
            _accepted("a", "Mouse"),
            _accepted("b", "Mouse", aligned_bases=128, fragment_count=2),
            {
                **_accepted("a", "Homo sapiens"),
                "alignment_source": "human_reference",
            },
        ]
    )
    summary = summarize_accepted_rows(
        rows.lazy(), policy="full_window", landmark_width=255
    ).row(0, named=True)
    assert summary["accepted_rows"] == 2
    assert summary["mean_fragment_count"] == 1.5
    assert summary["mean_aligned_bases"] == 191.5
    assert summary["min_sequence_length"] == 255
    assert summary["max_sequence_length"] == 255
    assert summary["plus_strand_rows"] == 2
    assert summary["minus_strand_rows"] == 0
    assert summary["negative_target_start_rows"] == 0
    assert abs(summary["mean_ambiguous_base_fraction"] - 0.2) < 1e-12
    assert abs(summary["mean_repeat_masked_fraction"] - 0.4) < 1e-12
    assert abs(summary["mean_gc_fraction"] - 0.4) < 1e-12
    assert summary["aligned_fraction_scope"].startswith("source_landmark_only")


def test_outcome_counts_close_requested_grid_by_policy_region_and_species() -> None:
    anchors = pl.DataFrame(
        {
            "query_name": ["a", "b", "c"],
            "region_label": ["cds", "cds", "enhancer"],
        }
    )
    species = pl.DataFrame(
        {
            "scientific_name": ["Mouse", "Cow"],
            "backend": ["zoonomia_cactus", "zoonomia_cactus"],
            "clade": ["mammals", "mammals"],
            "selected": [True, True],
        }
    )
    full = pl.DataFrame(
        [
            _accepted("a", "Mouse"),
            _accepted("c", "Cow", region="enhancer"),
        ]
    ).lazy()
    center = pl.DataFrame(
        [
            _accepted("a", "Mouse", aligned_bases=1),
            _accepted("b", "Mouse", aligned_bases=1),
            _accepted("c", "Cow", region="enhancer", aligned_bases=1),
        ]
    ).lazy()
    full_rejected = _rejected(
        [
            ("b", "Mouse", "span_too_long"),
            ("a", "Cow", "no_unique_locus"),
        ]
    )
    center_rejected = _rejected(
        [
            ("a", "Cow", "no_unique_locus"),
            ("c", "Mouse", "out_of_bounds"),
        ]
    )
    outcomes, summary = build_outcome_counts(
        anchors,
        species,
        {"full_window": full, "center_1": center},
        {"full_window": full_rejected, "center_1": center_rejected},
    )

    per_cell = outcomes.group_by("projection_policy", "region_label", "species").agg(
        pl.col("count").sum()
    )
    cds = per_cell.filter(pl.col("region_label") == "cds")
    enhancer = per_cell.filter(pl.col("region_label") == "enhancer")
    assert set(cds["count"]) == {2}
    assert set(enhancer["count"]) == {1}

    full_cds = summary.filter(
        (pl.col("projection_policy") == "full_window")
        & (pl.col("region_label") == "cds")
    ).row(0, named=True)
    assert full_cds["requested_pairs"] == 4
    assert full_cds["accepted_pairs"] == 1
    assert full_cds["rejected_pairs"] == 2
    assert full_cds["no_mapping_pairs"] == 1
    assert full_cds["mean_accepted_species_per_anchor"] == 0.5

    center_cds = summary.filter(
        (pl.col("projection_policy") == "center_1") & (pl.col("region_label") == "cds")
    ).row(0, named=True)
    assert center_cds["requested_pairs"] == 4
    assert center_cds["accepted_pairs"] == 2
    assert center_cds["rejected_pairs"] == 1
    assert center_cds["no_mapping_pairs"] == 1
