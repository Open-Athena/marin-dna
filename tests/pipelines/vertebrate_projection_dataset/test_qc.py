from __future__ import annotations

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.contract import (
    ACCEPTED_SCHEMA,
    REJECTION_SCHEMA,
)
from marin_dna.pipelines.vertebrate_projection_dataset.qc import (
    build_projection_qc_tables,
)
from tests.pipelines.vertebrate_projection_dataset.helpers import species_manifest


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
