"""Materialize the approved high-level AlphaGenome biosample taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


ISSUE = 421
EXPECTED_TRACKS = 4_430
EXPECTED_RAW_UNITS = 714
EXPECTED_TISSUE_GROUPS = 17
EXPECTED_CELL_LINEAGES = 12


TISSUE_LABELS = {
    "blood_immune": {
        "bodily fluid",
        "blood",
        "bone marrow",
        "immune organ",
        "lymph node",
        "lymphoid tissue",
        "spleen",
        "thymus",
    },
    "nervous_system": {"brain", "spinal cord", "nerve"},
    "cardiovascular": {
        "heart",
        "pericardium",
        "vasculature",
        "blood vessel",
        "arterial blood vessel",
        "vein",
        "lymphatic vessel",
    },
    "lung_airway": {"lung", "bronchus", "trachea", "nose", "mucous gland"},
    "liver": {"liver"},
    "gastrointestinal": {
        "intestine",
        "large intestine",
        "colon",
        "small intestine",
        "stomach",
        "esophagus",
        "mouth",
        "tongue",
        "gallbladder",
    },
    "pancreas": {"pancreas"},
    "kidney_urinary": {"kidney", "urinary bladder", "ureter"},
    "skin": {"skin of body", "skin of prepuce of penis", "hair follicle"},
    "breast_mammary": {"breast", "mammary gland"},
    "muscle": {"musculature of body"},
    "bone_connective": {"bone element", "skeleton", "connective tissue", "limb"},
    "adipose": {"adipose tissue"},
    "endocrine": {
        "adrenal gland",
        "thyroid gland",
        "paraythroid gland",
        "pituitary gland",
        "endocrine gland",
    },
    "reproductive": {
        "uterus",
        "vagina",
        "ovary",
        "gonad",
        "prostate gland",
        "testis",
        "penis",
    },
    "developmental_extraembryonic": {
        "embryo",
        "placenta",
        "extraembryonic component",
    },
    "eye": {"eye"},
}

# Specific organs win over generic ancestors such as endocrine gland or connective tissue.
TISSUE_PRIORITY = [
    "liver",
    "lung_airway",
    "kidney_urinary",
    "nervous_system",
    "cardiovascular",
    "breast_mammary",
    "gastrointestinal",
    "pancreas",
    "skin",
    "muscle",
    "blood_immune",
    "reproductive",
    "developmental_extraembryonic",
    "adipose",
    "endocrine",
    "eye",
    "bone_connective",
]

LINEAGE_LABELS = {
    "B_cell": {"B cell", "lymphoblast"},
    "T_cell": {"T cell", "CD4+ T cell", "CD8+ T cell"},
    "NK_cell": {"NK cell"},
    "myeloid": {"myeloid cell", "monocyte", "mononuclear cell"},
    "other_hematopoietic": {"hematopoietic cell", "leukocyte"},
    "epithelial": {"epithelial cell", "keratinocyte"},
    "fibroblast_stromal": {"fibroblast", "connective tissue cell", "pericyte"},
    "endothelial": {"endothelial cell"},
    "neural_glial": {"neural cell", "neuroblastoma cell"},
    "muscle_cardiac": {"smooth muscle cell", "myoblast", "cardiocyte"},
    "melanocyte": {"melanocyte"},
    "stem_progenitor": {
        "stem cell",
        "induced pluripotent stem cell",
        "embryonic cell",
        "progenitor cell",
    },
}

LINEAGE_PRIORITY = [
    "B_cell",
    "T_cell",
    "NK_cell",
    "myeloid",
    "other_hematopoietic",
    "endothelial",
    "epithelial",
    "fibroblast_stromal",
    "muscle_cardiac",
    "melanocyte",
    "neural_glial",
    "stem_progenitor",
]

TISSUE_OVERRIDES = {
    "UBERON:0000920": "developmental_extraembryonic",  # egg chorion
    "UBERON:0003729": "gastrointestinal",  # mouth mucosa
    "UBERON:0000057": "kidney_urinary",  # urethra
    "UBERON:0000998": "reproductive",  # seminal vesicle
    "UBERON:0001000": "reproductive",  # vas deferens
    "UBERON:0001154": "gastrointestinal",  # vermiform appendix
    "UBERON:0001301": "reproductive",  # epididymis
    "UBERON:0001736": "gastrointestinal",  # submandibular gland
    "UBERON:0001797": "eye",  # vitreous humor
    "UBERON:0001831": "gastrointestinal",  # parotid gland
    "UBERON:0002360": "nervous_system",  # meninx
    "UBERON:0002363": "nervous_system",  # dura mater
    "UBERON:0002448": "gastrointestinal",  # fungiform papilla
    "UBERON:0002581": "nervous_system",  # postcentral gyrus
    "UBERON:0002771": "nervous_system",  # middle temporal gyrus
    "UBERON:0003112": "nervous_system",  # olfactory region
    "UBERON:0003889": "reproductive",  # fallopian tube
    "UBERON:0005795": "reproductive",  # embryonic uterus
    "UBERON:0006659": "bone_connective",  # cruciate ligament
    "UBERON:0007190": "nervous_system",  # paracentral gyrus
    "UBERON:0008198": "skin",  # nail plate
    "EFO:0005913": "kidney_urinary",  # urothelium cell line
    "EFO:0000572": "blood_immune",  # lymphoblast entered as tissue
    "UBERON:0000458": "reproductive",  # endocervix
    "UBERON:0002394": "liver",  # bile duct
    "UBERON:0012249": "reproductive",  # ectocervix
    "EFO:0002819": "lung_airway",  # Calu3 lung cancer cell line
    "UBERON:0000341": "lung_airway",  # throat
    "UBERON:0001103": "muscle",  # diaphragm
    "UBERON:0001897": "nervous_system",  # thalamus
    "UBERON:0002134": "cardiovascular",  # tricuspid valve
    "UBERON:0002135": "cardiovascular",  # mitral valve
    "UBERON:0002146": "cardiovascular",  # pulmonary valve
    "UBERON:0002148": "nervous_system",  # locus ceruleus
    "UBERON:0002245": "nervous_system",  # cerebellar hemisphere
    "UBERON:0002336": "nervous_system",  # corpus callosum
}

LINEAGE_OVERRIDES = {
    "EFO:0005913": "epithelial",  # urothelium cell line
}

# Exact high-level ancestor classes used only when ENCODE slims are absent.
LINEAGE_ANCESTOR_IDS = {
    "B_cell": {"CLO:0000119"},  # immortal B cell line cell
}

TISSUE_NAME_PATTERNS = {
    "blood_immune": r"\b(blood|marrow|lymph|spleen|thymus)\b",
    "nervous_system": r"\b(brain|neural|neuron|astrocyte|spinal|nerve)\b",
    "cardiovascular": r"\b(heart|cardiac|artery|arterial|vein|vascular|endothelial)\b",
    "lung_airway": r"\b(lung|bronch|airway|trachea|nasal)\b",
    "liver": r"\b(liver|hepatic|hepatocyte)\b",
    "gastrointestinal": r"\b(colon|intestinal|intestine|stomach|gastric|esophag|rectal)\b",
    "pancreas": r"\b(pancrea)\w*\b",
    "kidney_urinary": r"\b(kidney|renal|bladder|ureter)\b",
    "skin": r"\b(skin|derm|foreskin|keratinocyte)\b",
    "breast_mammary": r"\b(breast|mammary)\b",
    "muscle": r"\b(muscle|myoblast|myotube)\b",
    "bone_connective": r"\b(bone|osteoblast|chondrocyte|cartilage)\b",
    "adipose": r"\b(adipose|adipocyte)\b",
    "endocrine": r"\b(adrenal|thyroid|pituitary|parathyroid)\b",
    "reproductive": r"\b(ovary|ovarian|uter|endometr|cervi|vagina|prostate|testis|testicular|penis)\b",
    "developmental_extraembryonic": r"\b(embryo|placenta|trophoblast)\b",
    "eye": r"\b(eye|retina|retinal|cornea)\b",
}

LINEAGE_NAME_PATTERNS = {
    "B_cell": r"\b(B cell|B-cell|lymphoblast)\b",
    "T_cell": r"\b(T cell|T-cell|CD4|CD8)\b",
    "NK_cell": r"\b(natural killer|NK cell)\b",
    "myeloid": r"\b(myeloid|monocyte|macrophage|neutrophil|dendritic)\b",
    "other_hematopoietic": r"\b(hematopoietic|leukocyte)\b",
    "epithelial": r"\b(epithelial|epithelium|keratinocyte)\b",
    "fibroblast_stromal": r"\b(fibroblast|stromal|pericyte)\b",
    "endothelial": r"\b(endothelial)\b",
    "neural_glial": r"\b(neural|neuron|astrocyte|glial|oligodendrocyte|neuroblast)\b",
    "muscle_cardiac": r"\b(muscle|myoblast|myotube|cardiomyocyte|cardiocyte)\b",
    "melanocyte": r"\b(melanocyte)\b",
    "stem_progenitor": r"\b(stem|progenitor|pluripotent|iPSC|embryonic)\b",
}


def candidates_from_labels(
    labels: list[str], mapping: dict[str, set[str]]
) -> list[str]:
    label_set = set(labels)
    return sorted(group for group, source in mapping.items() if label_set & source)


def candidates_from_name(name: str, patterns: dict[str, str]) -> list[str]:
    return sorted(
        group
        for group, pattern in patterns.items()
        if re.search(pattern, name, flags=re.IGNORECASE)
    )


def candidates_from_ancestor_ids(
    ancestor_ids: set[str], mapping: dict[str, set[str]]
) -> list[str]:
    return sorted(
        group for group, source_ids in mapping.items() if ancestor_ids & source_ids
    )


def choose(candidates: list[str], priority: list[str]) -> str | None:
    for group in priority:
        if group in candidates:
            return group
    return None


def classify_unit(
    unit: dict[str, Any],
    encode: dict[str, dict[str, Any]],
    ols: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    annotation = encode.get(str(unit["ontology_curie"]), {})
    fallback = ols.get(str(unit["ontology_curie"]), {})
    ancestor_labels = [
        str(value["label"])
        for value in fallback.get("ancestors", [])
        if value.get("label")
    ]
    fallback_label = fallback.get("label")
    if fallback_label:
        ancestor_labels.append(str(fallback_label))
    ancestor_ids = {
        str(value["obo_id"])
        for value in fallback.get("ancestors", [])
        if value.get("obo_id")
    }

    organ_labels = sorted(set(annotation.get("organ_slims", []) + ancestor_labels))
    cell_labels = sorted(
        set(annotation.get("cell_slims", []) + ancestor_labels) - {"cancer cell"}
    )
    semantic_text = " | ".join(
        [
            str(unit["canonical_biosample_name"]),
            *map(str, unit["alias_names"]),
        ]
    )

    tissue_candidates = candidates_from_labels(organ_labels, TISSUE_LABELS)
    tissue_source = "ontology_slim_or_ancestor"
    if not tissue_candidates:
        tissue_candidates = candidates_from_name(semantic_text, TISSUE_NAME_PATTERNS)
        tissue_source = "name_keyword_fallback" if tissue_candidates else None
    tissue_group = choose(tissue_candidates, TISSUE_PRIORITY)
    ontology_curie = str(unit["ontology_curie"])
    if ontology_curie in TISSUE_OVERRIDES:
        tissue_group = TISSUE_OVERRIDES[ontology_curie]
        tissue_candidates = sorted(set(tissue_candidates) | {tissue_group})
        tissue_source = "manual_ontology_review"

    lineage_candidates: list[str] = []
    lineage_source: str | None = None
    if unit["biosample_type"] != "tissue":
        lineage_candidates = candidates_from_labels(cell_labels, LINEAGE_LABELS)
        lineage_source = "ontology_slim_or_ancestor"
        if not lineage_candidates:
            lineage_candidates = candidates_from_ancestor_ids(
                ancestor_ids, LINEAGE_ANCESTOR_IDS
            )
            lineage_source = "ontology_ancestor_id" if lineage_candidates else None
        if not lineage_candidates:
            lineage_candidates = candidates_from_name(
                semantic_text, LINEAGE_NAME_PATTERNS
            )
            lineage_source = "name_keyword_fallback" if lineage_candidates else None
    if {
        "induced pluripotent stem cell",
        "embryonic cell",
    } & set(cell_labels):
        lineage_group = "stem_progenitor"
    else:
        lineage_group = choose(lineage_candidates, LINEAGE_PRIORITY)
    if ontology_curie in LINEAGE_OVERRIDES:
        lineage_group = LINEAGE_OVERRIDES[ontology_curie]
        lineage_candidates = sorted(set(lineage_candidates) | {lineage_group})
        lineage_source = "manual_ontology_review"

    return {
        **unit,
        "tissue_group": tissue_group,
        "tissue_source": tissue_source,
        "tissue_candidates": tissue_candidates,
        "cell_lineage": lineage_group,
        "cell_lineage_source": lineage_source,
        "cell_lineage_candidates": lineage_candidates,
        "encode_match": bool(annotation),
    }


def group_summary(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return (
        frame.filter(pl.col(column).is_not_null())
        .group_by(column)
        .agg(pl.col("track_count").sum().alias("tracks"), pl.len().alias("raw_units"))
        .sort("tracks", descending=True)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_taxonomy(
    units: pl.DataFrame,
    tracks: pl.DataFrame,
    encode_payload: dict[str, Any],
    ols_payload: dict[str, Any],
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, Any],
]:
    assert units.height == units["canonical_biosample_id"].n_unique()
    assert tracks.height == tracks["track_id"].n_unique()
    assert set(tracks["canonical_biosample_id"]) <= set(units["canonical_biosample_id"])
    encode = {
        item["term_id"]: item
        for item in encode_payload["@graph"]
        if item.get("term_id")
    }
    ols = ols_payload["records"]

    taxonomy = pl.DataFrame(
        [classify_unit(unit, encode, ols) for unit in units.iter_rows(named=True)],
        infer_schema_length=None,
    ).sort("track_count", "canonical_biosample_id", descending=[True, False])
    assert taxonomy.height == units.height
    assert taxonomy["canonical_biosample_id"].n_unique() == units.height

    track_taxonomy = tracks.join(
        taxonomy.select(
            "canonical_biosample_id",
            "tissue_group",
            "tissue_source",
            "cell_lineage",
            "cell_lineage_source",
        ),
        on="canonical_biosample_id",
        how="left",
        validate="m:1",
    )
    assert track_taxonomy.height == tracks.height
    assert track_taxonomy["track_id"].n_unique() == tracks.height

    tissue_summary = group_summary(taxonomy, "tissue_group")
    lineage_summary = group_summary(taxonomy, "cell_lineage")
    assay_tissue = (
        track_taxonomy.filter(pl.col("tissue_group").is_not_null())
        .group_by("assay", "tissue_group")
        .agg(
            pl.len().alias("track_count"),
            pl.col("track_id").sort().alias("track_ids"),
        )
        .sort("assay", "tissue_group")
        .with_columns(
            pl.concat_str("assay", "tissue_group", separator="|").alias("target_id")
        )
    )
    assay_lineage = (
        track_taxonomy.filter(pl.col("cell_lineage").is_not_null())
        .group_by("assay", "cell_lineage")
        .agg(
            pl.len().alias("track_count"),
            pl.col("track_id").sort().alias("track_ids"),
        )
        .sort("assay", "cell_lineage")
        .with_columns(
            pl.concat_str("assay", "cell_lineage", separator="|").alias("target_id")
        )
    )
    review = taxonomy.filter(
        (pl.col("tissue_candidates").list.len() > 1)
        | (pl.col("cell_lineage_candidates").list.len() > 1)
        | (pl.col("tissue_source") == "manual_ontology_review")
        | (pl.col("cell_lineage_source") == "manual_ontology_review")
        | (pl.col("tissue_group").is_null() & pl.col("cell_lineage").is_null())
    )

    tracks_with_tissue = int(track_taxonomy["tissue_group"].is_not_null().sum())
    tracks_with_lineage = int(track_taxonomy["cell_lineage"].is_not_null().sum())
    tracks_with_either = int(
        track_taxonomy.select(
            pl.any_horizontal(
                pl.col("tissue_group").is_not_null(),
                pl.col("cell_lineage").is_not_null(),
            ).sum()
        ).item()
    )
    unresolved = taxonomy.filter(
        pl.col("tissue_group").is_null() & pl.col("cell_lineage").is_null()
    )
    summary = {
        "raw_tracks": tracks.height,
        "raw_ontology_units": units.height,
        "tissue_groups": tissue_summary.height,
        "cell_lineages": lineage_summary.height,
        "assay_tissue_groups": assay_tissue.height,
        "assay_lineage_groups": assay_lineage.height,
        "tracks_with_tissue": tracks_with_tissue,
        "tracks_with_cell_lineage": tracks_with_lineage,
        "tracks_with_either": tracks_with_either,
        "tracks_unresolved_on_both_axes": tracks.height - tracks_with_either,
        "units_with_tissue": int(taxonomy["tissue_group"].is_not_null().sum()),
        "units_with_cell_lineage": int(taxonomy["cell_lineage"].is_not_null().sum()),
        "units_unresolved_on_both_axes": unresolved.height,
        "review_units": review.height,
        "manual_tissue_overrides": len(TISSUE_OVERRIDES),
        "manual_lineage_overrides": len(LINEAGE_OVERRIDES),
    }
    return (
        taxonomy,
        track_taxonomy,
        tissue_summary,
        lineage_summary,
        assay_tissue,
        assay_lineage,
        review,
        summary,
    )


def validate_production(
    taxonomy: pl.DataFrame,
    track_taxonomy: pl.DataFrame,
    summary: dict[str, Any],
) -> None:
    assert summary["raw_tracks"] == EXPECTED_TRACKS
    assert summary["raw_ontology_units"] == EXPECTED_RAW_UNITS
    assert summary["tissue_groups"] == EXPECTED_TISSUE_GROUPS
    assert summary["cell_lineages"] == EXPECTED_CELL_LINEAGES
    unresolved = taxonomy.filter(
        pl.col("tissue_group").is_null() & pl.col("cell_lineage").is_null()
    )
    assert unresolved.height == 1
    assert unresolved["ontology_curie"].item() == "UBERON:0007023"
    assert unresolved["canonical_biosample_name"].item() == "adult organism"
    assert unresolved["track_count"].item() == 2
    assert summary["tracks_with_either"] == EXPECTED_TRACKS - 2
    assert track_taxonomy.filter(pl.col("canonical_biosample_name") == "HepG2")[
        "tissue_group"
    ].unique().to_list() == ["liver"]
    assert track_taxonomy.filter(pl.col("canonical_biosample_name") == "GM12878")[
        "cell_lineage"
    ].unique().to_list() == ["B_cell"]


def render_audit(
    summary: dict[str, Any],
    tissue_summary: pl.DataFrame,
    lineage_summary: pl.DataFrame,
    unresolved: pl.DataFrame,
) -> str:
    lines = [
        "# AlphaGenome high-level biosample taxonomy audit",
        "",
        "This stage is metadata-only. It read no variant labels, AlphaGenome scores, or SAE activations.",
        "",
        "## Coverage",
        "",
        "- Tracks: {:,}".format(summary["raw_tracks"]),
        "- Exact ontology units: {:,}".format(summary["raw_ontology_units"]),
        "- Tissue groups: {}".format(summary["tissue_groups"]),
        "- Cell lineages: {}".format(summary["cell_lineages"]),
        "- Tracks assigned to at least one axis: {:,}".format(
            summary["tracks_with_either"]
        ),
        "- Tracks unresolved on both axes: {:,}".format(
            summary["tracks_unresolved_on_both_axes"]
        ),
        "",
        "## Tissue groups",
        "",
        tissue_summary.write_csv(),
        "## Cell lineages",
        "",
        lineage_summary.write_csv(),
        "## Unresolved on both axes",
        "",
        unresolved.select(
            "canonical_biosample_name",
            "biosample_type",
            "ontology_curie",
            "track_count",
        ).write_csv(),
    ]
    return "\n".join(lines).rstrip() + "\n"


def materialize(
    *,
    units_path: Path,
    tracks_path: Path,
    encode_path: Path,
    ols_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert all(
        path.is_file() for path in (units_path, tracks_path, encode_path, ols_path)
    )
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment_commit)

    units = pl.read_parquet(units_path)
    tracks = pl.read_parquet(tracks_path)
    encode_payload = json.loads(encode_path.read_text())
    ols_payload = json.loads(ols_path.read_text())
    (
        taxonomy,
        track_taxonomy,
        tissue_summary,
        lineage_summary,
        assay_tissue,
        assay_lineage,
        review,
        summary,
    ) = build_taxonomy(units, tracks, encode_payload, ols_payload)
    validate_production(taxonomy, track_taxonomy, summary)
    unresolved = taxonomy.filter(
        pl.col("tissue_group").is_null() & pl.col("cell_lineage").is_null()
    )

    output_dir.mkdir(parents=True)
    input_dir = output_dir / "inputs"
    input_dir.mkdir()
    input_copies = {
        "canonical_biosamples.parquet": units_path,
        "track_mapping.parquet": tracks_path,
        "encode_biosample_types.json": encode_path,
        "ols_fallback.json": ols_path,
    }
    for name, source in input_copies.items():
        shutil.copyfile(source, input_dir / name)

    outputs: dict[str, pl.DataFrame] = {
        "unit_taxonomy.parquet": taxonomy,
        "track_taxonomy.parquet": track_taxonomy,
        "tissue_summary.parquet": tissue_summary,
        "cell_lineage_summary.parquet": lineage_summary,
        "assay_tissue_groups.parquet": assay_tissue,
        "assay_cell_lineage_groups.parquet": assay_lineage,
        "review_queue.parquet": review,
    }
    artifacts: dict[str, dict[str, Any]] = {
        f"inputs/{name}": {
            "bytes": (input_dir / name).stat().st_size,
            "sha256": sha256_file(input_dir / name),
        }
        for name in input_copies
    }
    for name, frame in outputs.items():
        path = output_dir / name
        frame.write_parquet(path, compression="zstd")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": frame.height,
        }
    audit_path = output_dir / "AUDIT.md"
    audit_path.write_text(
        render_audit(summary, tissue_summary, lineage_summary, unresolved)
    )
    artifacts[audit_path.name] = {
        "bytes": audit_path.stat().st_size,
        "sha256": sha256_file(audit_path),
    }

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": output_dir.name,
        "experiment_commit": experiment_commit,
        "inputs": {
            "canonical_biosamples": {
                "bytes": units_path.stat().st_size,
                "sha256": sha256_file(units_path),
            },
            "track_mapping": {
                "bytes": tracks_path.stat().st_size,
                "sha256": sha256_file(tracks_path),
            },
            "encode_biosample_types": {
                "bytes": encode_path.stat().st_size,
                "sha256": sha256_file(encode_path),
                "records": len(encode_payload["@graph"]),
            },
            "ols_fallback": {
                "bytes": ols_path.stat().st_size,
                "sha256": sha256_file(ols_path),
                "records": len(ols_payload["records"]),
            },
        },
        "taxonomy": {
            "tissue_priority": TISSUE_PRIORITY,
            "lineage_priority": LINEAGE_PRIORITY,
            "tissue_overrides": TISSUE_OVERRIDES,
            "lineage_overrides": LINEAGE_OVERRIDES,
            "unresolved_policy": "exclude from the corresponding family; no heterogeneous other group",
        },
        "summary": summary,
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-biosamples", type=Path, required=True)
    parser.add_argument("--track-mapping", type=Path, required=True)
    parser.add_argument("--encode-biosample-types", type=Path, required=True)
    parser.add_argument("--ols-fallback", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        units_path=args.canonical_biosamples,
        tracks_path=args.track_mapping,
        encode_path=args.encode_biosample_types,
        ols_path=args.ols_fallback,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
