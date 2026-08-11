from __future__ import annotations

from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.contract import (
    ACCEPTED_SCHEMA,
    REJECTION_SCHEMA,
)
from marin_dna_vertebrate_projection.qc import (
    build_projection_qc_tables,
    write_projection_qc_tables_streaming,
)

from .helpers import species_manifest


def _accepted(
    query_name: str,
    species: str,
    alignment_source: str,
    clade: str,
    phylogenetic_rank: int,
) -> dict[str, object]:
    is_human = alignment_source == "human_reference"
    return {
        "query_name": query_name,
        "source_chrom": "chr1",
        "source_start": 0,
        "source_end": 255,
        "region_label": "cds",
        "species": species,
        "alignment_name": "hg38" if is_human else species.replace(" ", "_"),
        "assembly": "hg38" if is_human else species.replace(" ", "_assembly"),
        "taxonomy_id": 9606
        if is_human
        else {"Mus musculus": 10090, "Gallus gallus": 9031}[species],
        "family": "Hominidae"
        if is_human
        else {"Mus musculus": "Muridae", "Gallus gallus": "Phasianidae"}[species],
        "clade": clade,
        "phylogenetic_rank": phylogenetic_rank,
        "alignment_source": alignment_source,
        "t_chrom": "chr1",
        "t_start": 0,
        "t_end": 255,
        "t_strand": "+",
        "t_src_size": 1000,
        "pre_resize_t_start": 0,
        "pre_resize_t_end": 255,
        "fragment_count": 1,
        "aligned_bases": 255,
    }


def test_qc_reports_recovery_breadth_and_missing_reasons() -> None:
    anchors = pl.DataFrame(
        {
            "query_name": ["a1", "a2"],
            "source_chrom": ["chr1", "chr18"],
            "source_start": [0, 0],
            "source_end": [255, 255],
            "region_label": ["cds", "enhancer"],
            "split": ["train", "validation"],
        }
    )
    accepted = pl.DataFrame(
        [
            _accepted("a1", "Homo sapiens", "human_reference", "mammals", 0),
            _accepted("a1", "Mus musculus", "zoonomia_cactus", "mammals", 1),
            _accepted("a1", "Gallus gallus", "ucsc_multiz100way", "birds", 2),
        ],
        schema=ACCEPTED_SCHEMA,
    )
    rejected = pl.DataFrame(
        [
            {
                "query_name": "a1",
                "source_chrom": "chr1",
                "source_start": 0,
                "source_end": 255,
                "region_label": "cds",
                "species": "Xenopus tropicalis",
                "assembly": "JGI_7.0",
                "taxonomy_id": 8364,
                "family": "Pipidae",
                "clade": "amphibians",
                "phylogenetic_rank": 4,
                "alignment_source": "ucsc_multiz100way",
                "rejection_reason": "multi_strand",
                "detail": "+,-",
                "fragment_count": 2,
            }
        ],
        schema=REJECTION_SCHEMA,
    )
    tables = build_projection_qc_tables(anchors, accepted, rejected, species_manifest())
    a1 = tables.per_anchor.filter(pl.col("query_name") == "a1").row(0, named=True)
    assert a1["accepted_mammal_projections"] == 1
    assert a1["accepted_non_mammal_projections"] == 1
    assert a1["requested_total_species"] == 4
    assert a1["recovered_fraction"] == 0.5
    assert a1["deepest_recovered_clade"] == "birds"
    assert a1["no_mapping_count"] == 1
    reasons = tables.rejection_counts.filter(pl.col("query_name") == "a1")
    assert dict(reasons.select("rejection_reason", "count").iter_rows()) == {
        "multi_strand": 1,
        "no_mapping": 1,
    }
    assert set(tables.aggregates["region_label"].to_list()) == {
        "cds",
        "enhancer",
    }


def test_streaming_qc_matches_recovery_contract(tmp_path: Path) -> None:
    anchors = pl.DataFrame(
        {
            "query_name": ["a1", "a2"],
            "source_chrom": ["chr1", "chr18"],
            "source_start": [0, 0],
            "source_end": [255, 255],
            "region_label": ["cds", "enhancer"],
            "split": ["train", "validation"],
        }
    )
    accepted = pl.DataFrame(
        [
            _accepted("a1", "Homo sapiens", "human_reference", "mammals", 0),
            _accepted("a1", "Mus musculus", "zoonomia_cactus", "mammals", 1),
            _accepted("a1", "Gallus gallus", "ucsc_multiz100way", "birds", 2),
        ],
        schema=ACCEPTED_SCHEMA,
    )
    rejected = pl.DataFrame(
        [
            {
                "query_name": "a1",
                "source_chrom": "chr1",
                "source_start": 0,
                "source_end": 255,
                "region_label": "cds",
                "species": "Xenopus tropicalis",
                "assembly": "JGI_7.0",
                "taxonomy_id": 8364,
                "family": "Pipidae",
                "clade": "amphibians",
                "phylogenetic_rank": 4,
                "alignment_source": "ucsc_multiz100way",
                "rejection_reason": "multi_strand",
                "detail": "+,-",
                "fragment_count": 2,
            }
        ],
        schema=REJECTION_SCHEMA,
    )
    accepted_path = tmp_path / "accepted.parquet"
    rejected_path = tmp_path / "rejected.parquet"
    accepted.write_parquet(accepted_path)
    rejected.write_parquet(rejected_path)
    outputs = [tmp_path / f"qc-{index}.parquet" for index in range(4)]

    write_projection_qc_tables_streaming(
        anchors,
        accepted_path,
        [str(rejected_path)],
        species_manifest(),
        *outputs,
    )

    per_anchor = pl.read_parquet(outputs[0])
    a1 = per_anchor.filter(pl.col("query_name") == "a1").row(0, named=True)
    assert a1["accepted_mammal_projections"] == 1
    assert a1["accepted_non_mammal_projections"] == 1
    assert a1["requested_total_species"] == 4
    assert a1["recovered_fraction"] == 0.5
    assert a1["deepest_recovered_clade"] == "birds"
    assert a1["no_mapping_count"] == 1
    reasons = pl.read_parquet(outputs[2]).filter(pl.col("query_name") == "a1")
    assert dict(reasons.select("rejection_reason", "count").iter_rows()) == {
        "multi_strand": 1,
        "no_mapping": 1,
    }
    assert set(pl.read_parquet(outputs[3])["region_label"].to_list()) == {
        "cds",
        "enhancer",
    }
