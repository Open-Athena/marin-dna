"""Materialize the frozen AlphaGenome track-to-biosample ontology mapping.

This stage is metadata-only. It must not read AlphaGenome scores or SAE outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

ISSUE = 421
RUN_ID = "dna-exp421-biosample-mapping-r1"
EXPECTED_TRACKS = 4_430
EXPECTED_CANONICAL_UNITS = 714
EXPECTED_ASSAY_UNIT_GROUPS = 1_411
EXPECTED_ASSAYS = {
    "ATAC",
    "CAGE",
    "CHIP_HISTONE",
    "CHIP_TF",
    "DNASE",
    "PROCAP",
    "RNA_SEQ",
}
EXPECTED_TYPES = {
    "cell_line",
    "primary_cell",
    "tissue",
    "in_vitro_differentiated_cells",
    "organoid",
}
HIGH_FREQUENCY_TRACKS = 20
REQUIRED_COLUMNS = {
    "track_id",
    "assay",
    "assay_index",
    "name",
    "strand",
    "Assay title",
    "ontology_curie",
    "biosample_name",
    "biosample_type",
    "biosample_life_stage",
    "data_source",
    "endedness",
    "genetically_modified",
    "nonzero_mean",
    "transcription_factor",
    "histone_mark",
    "gtex_tissue",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank(values: list[object]) -> list[str]:
    return sorted(
        {
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        }
    )


def _canonical_name(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["biosample_name"]).strip() for row in rows)
    assert counts and all(counts)
    return min(counts, key=lambda name: (-counts[name], name.casefold(), name))


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    assert normalized
    return normalized


def _quantiles(values: list[int]) -> dict[str, float | int]:
    assert values
    series = pl.Series("value", values)
    return {
        "min": min(values),
        "q25": float(series.quantile(0.25, interpolation="linear")),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75, interpolation="linear")),
        "q90": float(series.quantile(0.90, interpolation="linear")),
        "q99": float(series.quantile(0.99, interpolation="linear")),
        "max": max(values),
    }


def build_mapping(
    metadata: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Build exact semantic-class + ontology-CURIE mapping artifacts."""

    assert REQUIRED_COLUMNS <= set(metadata.columns)
    assert metadata.height == metadata["track_id"].n_unique()
    assert not metadata.filter(
        pl.col("track_id").is_null()
        | (pl.col("track_id").str.strip_chars() == "")
        | pl.col("ontology_curie").is_null()
        | (pl.col("ontology_curie").str.strip_chars() == "")
        | pl.col("biosample_name").is_null()
        | (pl.col("biosample_name").str.strip_chars() == "")
        | pl.col("biosample_type").is_null()
        | (pl.col("biosample_type").str.strip_chars() == "")
    ).height

    rows = metadata.sort("track_id").to_dicts()
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    types_by_curie: dict[str, set[str]] = defaultdict(set)
    curies_by_normalized_name: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        biosample_type = str(row["biosample_type"])
        ontology_curie = str(row["ontology_curie"])
        canonical_id = f"{biosample_type}|{ontology_curie}"
        by_unit[canonical_id].append(row)
        types_by_curie[ontology_curie].add(biosample_type)
        curies_by_normalized_name[_normalize_name(str(row["biosample_name"]))].add(
            ontology_curie
        )

    cross_curie_name_collisions = {
        name: sorted(curies)
        for name, curies in curies_by_normalized_name.items()
        if len(curies) > 1
    }
    assert not cross_curie_name_collisions, cross_curie_name_collisions

    unit_records: list[dict[str, Any]] = []
    unit_lookup: dict[str, dict[str, Any]] = {}
    for canonical_id, unit_rows in sorted(by_unit.items()):
        biosample_type = str(unit_rows[0]["biosample_type"])
        ontology_curie = str(unit_rows[0]["ontology_curie"])
        assert all(str(row["biosample_type"]) == biosample_type for row in unit_rows)
        assert all(str(row["ontology_curie"]) == ontology_curie for row in unit_rows)
        aliases = sorted({str(row["biosample_name"]) for row in unit_rows})
        assays = sorted({str(row["assay"]) for row in unit_rows})
        sources = sorted({str(row["data_source"]) for row in unit_rows})
        record = {
            "canonical_biosample_id": canonical_id,
            "canonical_biosample_name": _canonical_name(unit_rows),
            "biosample_type": biosample_type,
            "ontology_curie": ontology_curie,
            "ontology_prefix": ontology_curie.split(":", maxsplit=1)[0],
            "alias_names": aliases,
            "alias_count": len(aliases),
            "track_count": len(unit_rows),
            "assays": assays,
            "assay_count": len(assays),
            "data_sources": sources,
            "data_source_count": len(sources),
            "life_stages": _nonblank(
                [row["biosample_life_stage"] for row in unit_rows]
            ),
            "gtex_tissues": _nonblank([row["gtex_tissue"] for row in unit_rows]),
            "ontology_curie_spans_semantic_classes": (
                len(types_by_curie[ontology_curie]) > 1
            ),
            "mapping_status": (
                "ontology_exact_synonym_merged"
                if len(aliases) > 1
                else "ontology_exact"
            ),
        }
        unit_records.append(record)
        unit_lookup[canonical_id] = record

    track_records: list[dict[str, Any]] = []
    assay_unit_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        canonical_id = f"{row['biosample_type']}|{row['ontology_curie']}"
        unit = unit_lookup[canonical_id]
        assay_unit_rows[(str(row["assay"]), canonical_id)].append(row)
        track_records.append(
            {
                **row,
                "canonical_biosample_id": canonical_id,
                "canonical_biosample_name": unit["canonical_biosample_name"],
                "canonical_biosample_type": unit["biosample_type"],
                "ontology_prefix": unit["ontology_prefix"],
                "mapping_status": unit["mapping_status"],
                "ontology_curie_spans_semantic_classes": unit[
                    "ontology_curie_spans_semantic_classes"
                ],
                "track_count_in_canonical_unit": unit["track_count"],
            }
        )

    assay_unit_records: list[dict[str, Any]] = []
    assay_unit_count: dict[tuple[str, str], int] = {}
    for (assay, canonical_id), group_rows in sorted(assay_unit_rows.items()):
        unit = unit_lookup[canonical_id]
        assay_unit_count[(assay, canonical_id)] = len(group_rows)
        assay_unit_records.append(
            {
                "assay_biosample_id": f"{assay}|{canonical_id}",
                "assay": assay,
                "canonical_biosample_id": canonical_id,
                "canonical_biosample_name": unit["canonical_biosample_name"],
                "biosample_type": unit["biosample_type"],
                "ontology_curie": unit["ontology_curie"],
                "track_count": len(group_rows),
                "track_ids": sorted(str(row["track_id"]) for row in group_rows),
                "strands": _nonblank([row["strand"] for row in group_rows]),
                "transcription_factors": _nonblank(
                    [row["transcription_factor"] for row in group_rows]
                ),
                "histone_marks": _nonblank([row["histone_mark"] for row in group_rows]),
                "endedness": _nonblank([row["endedness"] for row in group_rows]),
                "data_sources": _nonblank([row["data_source"] for row in group_rows]),
            }
        )

    track_mapping = pl.DataFrame(track_records).with_columns(
        pl.struct("assay", "canonical_biosample_id")
        .map_elements(
            lambda value: assay_unit_count[
                (str(value["assay"]), str(value["canonical_biosample_id"]))
            ],
            return_dtype=pl.UInt16,
        )
        .alias("track_count_in_assay_unit")
    )
    unit_catalog = pl.DataFrame(unit_records).sort("canonical_biosample_id")
    assay_unit_catalog = pl.DataFrame(assay_unit_records).sort(
        "assay", "canonical_biosample_id"
    )

    review_records: list[dict[str, Any]] = []
    for record in unit_records:
        reasons: list[str] = []
        decisions: list[str] = []
        if bool(record["ontology_curie_spans_semantic_classes"]):
            reasons.append("ontology_curie_spans_semantic_classes")
            decisions.append("keep_separate_by_biosample_type")
        if int(record["alias_count"]) > 1:
            reasons.append("same_curie_multiple_labels")
            decisions.append("merge_labels_within_type_and_curie")
        if record["ontology_prefix"] == "NTR":
            reasons.append("nonstandard_ntr_identifier")
            decisions.append("retain_exact_unresolved_identifier")
        if int(record["track_count"]) >= HIGH_FREQUENCY_TRACKS:
            reasons.append("high_frequency_unit")
            decisions.append("retain_and_report_extreme_value_group_size")
        if reasons:
            review_records.append(
                {
                    **record,
                    "review_reasons": reasons,
                    "review_decisions": decisions,
                    "review_status": "reviewed_before_outcome_aggregation",
                }
            )
    review_queue = pl.DataFrame(review_records).sort(
        "track_count", "canonical_biosample_id", descending=[True, False]
    )

    assert track_mapping.height == metadata.height
    assert track_mapping["track_id"].n_unique() == metadata.height
    assert unit_catalog["canonical_biosample_id"].n_unique() == unit_catalog.height
    assert (
        assay_unit_catalog["assay_biosample_id"].n_unique() == assay_unit_catalog.height
    )
    assert assay_unit_catalog["track_count"].sum() == metadata.height
    assert set(track_mapping["canonical_biosample_id"].to_list()) <= set(
        unit_catalog["canonical_biosample_id"].to_list()
    )
    assert (
        track_mapping.select(
            pl.struct("assay", "canonical_biosample_id").n_unique()
        ).item()
        == assay_unit_catalog.height
    )

    counts_by_type = {
        str(biosample_type): int(count)
        for biosample_type, count in unit_catalog.group_by("biosample_type")
        .len()
        .sort("biosample_type")
        .iter_rows()
    }
    assay_counts = {
        str(assay): {"tracks": int(tracks), "canonical_units": int(units)}
        for assay, tracks, units in track_mapping.group_by("assay")
        .agg(
            pl.len().alias("tracks"),
            pl.col("canonical_biosample_id").n_unique().alias("units"),
        )
        .sort("assay")
        .iter_rows()
    }
    summary = {
        "tracks": metadata.height,
        "canonical_units": unit_catalog.height,
        "assay_unit_groups": assay_unit_catalog.height,
        "outcome_groups": {
            "global": 1,
            "assay": len(assay_counts),
            "biosample": unit_catalog.height,
            "assay_biosample": assay_unit_catalog.height,
        },
        "units_by_type": counts_by_type,
        "assays": assay_counts,
        "unit_track_count_distribution": _quantiles(
            [int(value) for value in unit_catalog["track_count"].to_list()]
        ),
        "assay_unit_track_count_distribution": _quantiles(
            [int(value) for value in assay_unit_catalog["track_count"].to_list()]
        ),
        "singleton_units": int(unit_catalog.filter(pl.col("track_count") == 1).height),
        "multi_assay_units": int(unit_catalog.filter(pl.col("assay_count") > 1).height),
        "curies_spanning_semantic_classes": sum(
            len(types) > 1 for types in types_by_curie.values()
        ),
        "canonical_units_with_aliases": int(
            unit_catalog.filter(pl.col("alias_count") > 1).height
        ),
        "ntr_units": int(
            unit_catalog.filter(pl.col("ontology_prefix") == "NTR").height
        ),
        "high_frequency_units": int(
            unit_catalog.filter(pl.col("track_count") >= HIGH_FREQUENCY_TRACKS).height
        ),
        "reviewed_units": review_queue.height,
        "normalized_name_cross_curie_collisions": 0,
    }
    return track_mapping, unit_catalog, assay_unit_catalog, review_queue, summary


def validate_production(metadata: pl.DataFrame, summary: dict[str, Any]) -> None:
    assert metadata.height == EXPECTED_TRACKS
    assert set(metadata["assay"].unique().to_list()) == EXPECTED_ASSAYS
    assert set(metadata["biosample_type"].unique().to_list()) == EXPECTED_TYPES
    assert summary["canonical_units"] == EXPECTED_CANONICAL_UNITS
    assert summary["assay_unit_groups"] == EXPECTED_ASSAY_UNIT_GROUPS


def render_audit(
    unit_catalog: pl.DataFrame,
    review_queue: pl.DataFrame,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# exp421 AlphaGenome biosample ontology audit",
        "",
        "This artifact was built from AlphaGenome output metadata only. It did not read L2 scores, SAE activations, labels, or subsets.",
        "",
        "## Frozen mapping",
        "",
        "`canonical_biosample_id = biosample_type + '|' + ontology_curie`.",
        "",
        f"- Tracks: {summary['tracks']:,}",
        f"- Canonical biosamples: {summary['canonical_units']:,}",
        f"- Assay × biosample groups: {summary['assay_unit_groups']:,}",
        f"- Reviewed units: {summary['reviewed_units']:,}",
        f"- CURIEs spanning semantic classes: {summary['curies_spanning_semantic_classes']:,}",
        f"- Units with label aliases: {summary['canonical_units_with_aliases']:,}",
        f"- NTR units retained unresolved: {summary['ntr_units']:,}",
        "",
        "## Unit classes",
        "",
        "| semantic class | units |",
        "|---|---:|",
    ]
    for key, value in summary["units_by_type"].items():
        lines.append(f"| {key} | {value:,} |")
    lines.extend(
        [
            "",
            "## Assay membership",
            "",
            "| assay | tracks | canonical units |",
            "|---|---:|---:|",
        ]
    )
    for assay, counts in summary["assays"].items():
        lines.append(
            f"| {assay} | {counts['tracks']:,} | {counts['canonical_units']:,} |"
        )
    lines.extend(
        [
            "",
            "## Highest-frequency units",
            "",
            "| canonical unit | display name | tracks | assays |",
            "|---|---|---:|---:|",
        ]
    )
    for row in (
        unit_catalog.sort("track_count", descending=True).head(25).iter_rows(named=True)
    ):
        lines.append(
            f"| `{row['canonical_biosample_id']}` | {row['canonical_biosample_name']} | "
            f"{row['track_count']} | {row['assay_count']} |"
        )
    lines.extend(
        [
            "",
            "## Review policy",
            "",
            "Cross-class CURIEs remain separated by biosample type; same-type/same-CURIE aliases merge; NTR identifiers remain exact and unresolved; high-frequency units remain included with their group size reported. See `review_queue.parquet` for every reviewed unit and decision.",
            "",
            f"Review queue rows: {review_queue.height:,}.",
            "",
        ]
    )
    return "\n".join(lines)


def materialize(
    *,
    metadata_path: Path,
    metadata_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40 and all(
        character in "0123456789abcdef" for character in experiment_commit
    )
    source_manifest = json.loads(metadata_manifest_path.read_text())
    assert source_manifest["rows"] == EXPECTED_TRACKS
    assert sha256_file(metadata_path) == source_manifest["artifact"]["sha256"]

    metadata = pl.read_parquet(metadata_path)
    mapping, units, assay_units, review, summary = build_mapping(metadata)
    validate_production(metadata, summary)

    output_dir.mkdir(parents=True)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir()
    shutil.copyfile(metadata_manifest_path, inputs_dir / "track_metadata_manifest.json")
    mapping.write_parquet(output_dir / "track_mapping.parquet", compression="zstd")
    units.write_parquet(output_dir / "canonical_biosamples.parquet", compression="zstd")
    assay_units.write_parquet(
        output_dir / "assay_biosample_groups.parquet", compression="zstd"
    )
    review.write_parquet(output_dir / "review_queue.parquet", compression="zstd")
    (output_dir / "AUDIT.md").write_text(render_audit(units, review, summary))

    artifacts = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "issue": ISSUE,
        "run_id": RUN_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": experiment_commit,
        "analysis_status": "metadata_only_mapping_frozen_before_outcome_analysis",
        "outcome_data_read": False,
        "mapping_rule": "biosample_type|ontology_curie",
        "input": {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "metadata_manifest_sha256": sha256_file(metadata_manifest_path),
            "alphagenome_version": source_manifest["alphagenome_version"],
        },
        "summary": summary,
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        metadata_path=args.metadata,
        metadata_manifest_path=args.metadata_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
