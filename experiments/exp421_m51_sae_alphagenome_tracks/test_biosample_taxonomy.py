from __future__ import annotations

import polars as pl

from biosample_taxonomy import build_taxonomy, classify_unit


def _unit(
    name: str,
    *,
    curie: str,
    biosample_type: str = "cell_line",
    aliases: list[str] | None = None,
    track_count: int = 1,
) -> dict[str, object]:
    return {
        "canonical_biosample_id": f"{biosample_type}|{curie}",
        "canonical_biosample_name": name,
        "alias_names": aliases or [name],
        "biosample_type": biosample_type,
        "ontology_curie": curie,
        "track_count": track_count,
    }


def test_classification_uses_specific_tissue_and_lineage_priorities() -> None:
    encode = {
        "EFO:HEPG2": {
            "organ_slims": ["endocrine gland", "liver"],
            "cell_slims": ["epithelial cell"],
        },
        "EFO:GM12878": {
            "organ_slims": ["blood"],
            "cell_slims": ["hematopoietic cell", "B cell"],
        },
        "EFO:WTC11": {
            "organ_slims": ["skin of body"],
            "cell_slims": ["fibroblast", "induced pluripotent stem cell"],
        },
        "CL:RPE": {
            "organ_slims": ["eye"],
            "cell_slims": ["epithelial cell", "neural cell"],
        },
    }

    hepg2 = classify_unit(_unit("HepG2", curie="EFO:HEPG2"), encode, {})
    gm12878 = classify_unit(_unit("GM12878", curie="EFO:GM12878"), encode, {})
    wtc11 = classify_unit(_unit("WTC11", curie="EFO:WTC11"), encode, {})
    retinal_epithelium = classify_unit(
        _unit("retinal pigment epithelial cell", curie="CL:RPE"), encode, {}
    )

    assert (hepg2["tissue_group"], hepg2["cell_lineage"]) == (
        "liver",
        "epithelial",
    )
    assert gm12878["cell_lineage"] == "B_cell"
    assert wtc11["cell_lineage"] == "stem_progenitor"
    assert retinal_epithelium["cell_lineage"] == "epithelial"


def test_classification_applies_reviewed_ontology_overrides() -> None:
    result = classify_unit(_unit("urothelium cell line", curie="EFO:0005913"), {}, {})
    assert result["tissue_group"] == "kidney_urinary"
    assert result["tissue_source"] == "manual_ontology_review"
    assert result["cell_lineage"] == "epithelial"
    assert result["cell_lineage_source"] == "manual_ontology_review"


def test_keyword_fallback_does_not_match_noisy_ancestor_substrings() -> None:
    ols = {
        "CL:0000569": {
            "label": "cardiac mesenchymal cell",
            "ancestors": [{"label": "neural cell"}],
        }
    }
    result = classify_unit(
        _unit("cardiac mesenchymal cell", curie="CL:0000569"), {}, ols
    )
    assert result["tissue_group"] == "cardiovascular"
    assert result["tissue_candidates"] == ["cardiovascular"]


def test_reviewed_ancestor_id_classifies_lymphoblastoid_cell_lines() -> None:
    ols = {
        "CLO:0013950": {
            "label": "GM21619 cell",
            "ancestors": [
                {
                    "label": "immortal B cell line cell",
                    "obo_id": "CLO:0000119",
                }
            ],
        }
    }
    result = classify_unit(_unit("GM21619", curie="CLO:0013950"), {}, ols)
    assert result["cell_lineage"] == "B_cell"
    assert result["cell_lineage_source"] == "ontology_ancestor_id"


def test_build_taxonomy_preserves_tracks_and_builds_both_axes() -> None:
    units = pl.DataFrame(
        [
            _unit("HepG2", curie="EFO:HEPG2", track_count=2),
            _unit(
                "adult organism",
                curie="UBERON:0007023",
                biosample_type="tissue",
            ),
        ]
    )
    tracks = pl.DataFrame(
        [
            {
                "track_id": "ATAC_0",
                "canonical_biosample_id": "cell_line|EFO:HEPG2",
                "canonical_biosample_name": "HepG2",
                "assay": "ATAC",
            },
            {
                "track_id": "DNASE_1",
                "canonical_biosample_id": "cell_line|EFO:HEPG2",
                "canonical_biosample_name": "HepG2",
                "assay": "DNASE",
            },
            {
                "track_id": "ATAC_2",
                "canonical_biosample_id": "tissue|UBERON:0007023",
                "canonical_biosample_name": "adult organism",
                "assay": "ATAC",
            },
        ]
    )
    encode = {
        "@graph": [
            {
                "term_id": "EFO:HEPG2",
                "organ_slims": ["liver"],
                "cell_slims": ["epithelial cell"],
            }
        ]
    }

    (
        taxonomy,
        track_taxonomy,
        tissue_summary,
        lineage_summary,
        assay_tissue,
        assay_lineage,
        _,
        summary,
    ) = build_taxonomy(units, tracks, encode, {"records": {}})

    assert taxonomy.height == 2
    assert track_taxonomy.height == 3
    assert track_taxonomy["track_id"].n_unique() == 3
    assert tissue_summary.to_dicts() == [
        {"tissue_group": "liver", "tracks": 2, "raw_units": 1}
    ]
    assert lineage_summary.to_dicts() == [
        {"cell_lineage": "epithelial", "tracks": 2, "raw_units": 1}
    ]
    assert assay_tissue.height == 2
    assert assay_lineage.height == 2
    assert summary["tracks_with_either"] == 2
    assert summary["tracks_unresolved_on_both_axes"] == 1
