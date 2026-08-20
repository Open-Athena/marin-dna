from __future__ import annotations

import polars as pl
from marin_dna_vertebrate_projection.issue_473.fixed_catalog_report import (
    build_fixed_catalog_outcome_counts,
)


def _accepted() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "query_name": ["fixed-anchor"],
            "species": ["Mouse"],
            "region_label": ["cds"],
            "alignment_source": ["zoonomia_cactus"],
            "clade": ["mammals"],
        }
    ).lazy()


def _rejected(rows: list[str]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "query_name": rows,
            "species": ["Mouse"] * len(rows),
            "rejection_reason": ["no_unique_locus"] * len(rows),
        },
        schema={
            "query_name": pl.String,
            "species": pl.String,
            "rejection_reason": pl.String,
        },
    ).lazy()


def test_fixed_catalog_report_drops_only_superset_rejection_rows() -> None:
    anchors = pl.DataFrame({"query_name": ["fixed-anchor"], "region_label": ["cds"]})
    species = pl.DataFrame(
        {
            "scientific_name": ["Mouse"],
            "backend": ["zoonomia_cactus"],
            "clade": ["mammals"],
            "selected": [True],
        }
    )
    outcomes, summary = build_fixed_catalog_outcome_counts(
        anchors,
        species,
        {"full_window": _accepted(), "center_1": _accepted()},
        {
            "full_window": _rejected(["outside-anchor"]),
            "center_1": _rejected([]),
        },
    )
    assert outcomes["region_label"].null_count() == 0
    assert summary["region_label"].null_count() == 0
    assert summary.height == 2
    assert set(summary["accepted_pairs"]) == {1}
    assert set(summary["requested_pairs"]) == {1}
