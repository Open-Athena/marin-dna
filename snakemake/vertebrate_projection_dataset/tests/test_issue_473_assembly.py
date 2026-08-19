from __future__ import annotations

from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.issue_473.assembly import (
    write_full_window_qc_union,
    write_full_window_sequence_union,
    write_intersection_validation_views,
)


def _sequence(
    query: str, region: str, species: str = "Mus musculus"
) -> dict[str, object]:
    return {
        "query_name": query,
        "species": species,
        "source_chrom": "chr18",
        "source_start": 0,
        "source_end": 255,
        "region_label": region,
        "alignment_source": "zoonomia_cactus",
        "sequence": "A" * 255,
    }


def test_full_window_union_reuses_standard_and_replaces_old_ccre(
    tmp_path: Path,
) -> None:
    baseline = pl.DataFrame(
        [
            _sequence("cds", "cds"),
            _sequence("utr", "utr3"),
            _sequence("old-enhancer", "ccre_non_promoter"),
            _sequence("background", "background"),
        ]
    )
    enhancer = pl.DataFrame([_sequence("new-enhancer", "ccre_enhancer_centered")])
    baseline_path = tmp_path / "baseline.parquet"
    enhancer_path = tmp_path / "enhancer.parquet"
    output_path = tmp_path / "combined.parquet"
    baseline.write_parquet(baseline_path)
    enhancer.write_parquet(enhancer_path)

    write_full_window_sequence_union(baseline_path, enhancer_path, output_path)
    result = pl.read_parquet(output_path)

    assert set(result["query_name"]) == {"cds", "utr", "new-enhancer"}
    assert "old-enhancer" not in set(result["query_name"])


def test_qc_union_filters_baseline_regions(tmp_path: Path) -> None:
    baseline = pl.DataFrame({"region_label": ["cds", "background"], "count": [2, 3]})
    enhancer = pl.DataFrame({"region_label": ["ccre_enhancer_centered"], "count": [5]})
    baseline_path = tmp_path / "baseline-qc.parquet"
    enhancer_path = tmp_path / "enhancer-qc.parquet"
    output_path = tmp_path / "qc.parquet"
    baseline.write_parquet(baseline_path)
    enhancer.write_parquet(enhancer_path)

    write_full_window_qc_union(baseline_path, enhancer_path, output_path)
    assert pl.read_parquet(output_path)["count"].to_list() == [2, 5]


def test_intersection_validation_uses_identical_biological_rows(
    tmp_path: Path,
) -> None:
    full = pl.DataFrame(
        [
            _sequence("a1", "cds"),
            _sequence("a2", "cds"),
            _sequence("full-only", "cds"),
        ]
    )
    center = pl.DataFrame(
        [
            _sequence("a1", "cds"),
            _sequence("a2", "cds"),
            _sequence("center-only", "cds"),
        ]
    ).with_columns(pl.lit("C" * 255).alias("sequence"))
    full_path = tmp_path / "full.parquet"
    center_path = tmp_path / "center.parquet"
    full.write_parquet(full_path)
    center.write_parquet(center_path)

    full_output = tmp_path / "intersection-full.parquet"
    center_output = tmp_path / "intersection-center.parquet"
    selection = tmp_path / "selection.tsv"
    summary = tmp_path / "summary.json"
    write_intersection_validation_views(
        full_path,
        center_path,
        full_output,
        center_output,
        selection,
        summary,
        region_label="cds",
        max_validation_rows=2,
    )

    full_rows = pl.read_parquet(full_output)
    center_rows = pl.read_parquet(center_output)
    assert full_rows["row_id"].to_list() == center_rows["row_id"].to_list()
    assert set(full_rows["query_name"]) == {"a1", "a2"}
    assert full_rows["sequence"].to_list() != center_rows["sequence"].to_list()
