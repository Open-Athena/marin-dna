from __future__ import annotations

import polars as pl
import pytest

from biosample_mapping import build_mapping


def _row(
    track_id: str,
    *,
    assay: str,
    curie: str,
    name: str,
    biosample_type: str,
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "assay": assay,
        "assay_index": int(track_id.rsplit("_", maxsplit=1)[1]),
        "name": f"raw {track_id}",
        "strand": ".",
        "Assay title": assay,
        "ontology_curie": curie,
        "biosample_name": name,
        "biosample_type": biosample_type,
        "biosample_life_stage": None,
        "data_source": "encode",
        "endedness": None,
        "genetically_modified": False,
        "nonzero_mean": 1.0,
        "transcription_factor": None,
        "histone_mark": None,
        "gtex_tissue": None,
    }


def test_mapping_merges_synonyms_but_keeps_semantic_classes_separate() -> None:
    metadata = pl.DataFrame(
        [
            _row(
                "ATAC_0",
                assay="ATAC",
                curie="CL:1",
                name="adipocyte",
                biosample_type="primary_cell",
            ),
            _row(
                "ATAC_1",
                assay="ATAC",
                curie="CL:1",
                name="adipocyte",
                biosample_type="primary_cell",
            ),
            _row(
                "DNASE_2",
                assay="DNASE",
                curie="CL:1",
                name="fat cell",
                biosample_type="primary_cell",
            ),
            _row(
                "DNASE_3",
                assay="DNASE",
                curie="CL:1",
                name="adipocyte",
                biosample_type="in_vitro_differentiated_cells",
            ),
            _row(
                "RNA_SEQ_4",
                assay="RNA_SEQ",
                curie="NTR:2",
                name="unresolved tissue",
                biosample_type="tissue",
            ),
        ]
    )
    mapping, units, assay_units, review, summary = build_mapping(metadata)

    assert mapping.height == 5
    assert units.height == 3
    assert assay_units.height == 4
    assert summary["curies_spanning_semantic_classes"] == 1
    assert summary["canonical_units_with_aliases"] == 1
    assert summary["ntr_units"] == 1
    primary = units.filter(pl.col("canonical_biosample_id") == "primary_cell|CL:1").row(
        0, named=True
    )
    assert primary["canonical_biosample_name"] == "adipocyte"
    assert primary["alias_names"] == ["adipocyte", "fat cell"]
    assert primary["track_count"] == 3
    assert primary["ontology_curie_spans_semantic_classes"]
    assert set(review["canonical_biosample_id"].to_list()) == {
        "primary_cell|CL:1",
        "in_vitro_differentiated_cells|CL:1",
        "tissue|NTR:2",
    }
    assert (
        mapping.filter(pl.col("track_id") == "ATAC_0")[
            "track_count_in_assay_unit"
        ].item()
        == 2
    )


def test_mapping_rejects_same_normalized_name_for_distinct_curies() -> None:
    metadata = pl.DataFrame(
        [
            _row(
                "ATAC_0",
                assay="ATAC",
                curie="CL:1",
                name="B cell",
                biosample_type="primary_cell",
            ),
            _row(
                "DNASE_1",
                assay="DNASE",
                curie="CL:2",
                name="B-cell",
                biosample_type="primary_cell",
            ),
        ]
    )
    with pytest.raises(AssertionError, match="bcell"):
        build_mapping(metadata)
