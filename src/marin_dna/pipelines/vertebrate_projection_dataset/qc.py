"""Machine-readable per-anchor and aggregate projection QC tables."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.manifest import (
    validate_species_manifest,
)


@dataclass(frozen=True)
class ProjectionQcTables:
    """Normalized QC outputs written by the pipeline before upload."""

    per_anchor: pl.DataFrame
    per_anchor_scope: pl.DataFrame
    rejection_counts: pl.DataFrame
    aggregates: pl.DataFrame


def _as_int(value: object) -> int:
    assert isinstance(value, int | str) and not isinstance(value, bool)
    return int(value)


def _records_by_query(frame: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
    if frame.is_empty():
        return {}
    return {
        str(key[0] if isinstance(key, tuple) else key): group.to_dicts()
        for key, group in frame.partition_by("query_name", as_dict=True).items()
    }


def build_projection_qc_tables(
    anchors: pl.DataFrame,
    accepted: pl.DataFrame,
    rejected: pl.DataFrame,
    species_manifest: pl.DataFrame,
) -> ProjectionQcTables:
    """Build per-anchor recovery, scoped coverage, rejection, and summaries."""
    required_anchors = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "split",
    }
    missing = required_anchors - set(anchors.columns)
    assert not missing, f"QC anchors missing columns: {sorted(missing)}"
    assert anchors["query_name"].n_unique() == anchors.height
    validate_species_manifest(species_manifest)
    targets = species_manifest.filter(pl.col("selected"))
    target_species = set(targets["scientific_name"].to_list())

    accepted_targets = accepted.filter(pl.col("alignment_source") != "human_reference")
    assert set(accepted_targets["species"].to_list()) <= target_species
    assert accepted_targets.select("query_name", "species").is_unique().all()
    assert rejected.select("query_name", "species").is_unique().all()
    accepted_keys = set(accepted_targets.select("query_name", "species").iter_rows())
    rejected_keys = set(rejected.select("query_name", "species").iter_rows())
    assert accepted_keys.isdisjoint(rejected_keys)

    accepted_by_query = _records_by_query(accepted_targets)
    rejected_by_query = _records_by_query(rejected)
    scopes = [
        {
            "backend": str(row["backend"]),
            "clade": str(row["clade"]),
            "requested_species": int(row["len"]),
        }
        for row in targets.group_by("backend", "clade").len().to_dicts()
    ]
    requested_total = targets.height
    requested_mammals = targets.filter(pl.col("clade") == "mammals").height
    requested_non_mammals = requested_total - requested_mammals

    per_anchor_rows: list[dict[str, object]] = []
    scope_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    for anchor in anchors.to_dicts():
        query_name = str(anchor["query_name"])
        kept = accepted_by_query.get(query_name, [])
        dropped = rejected_by_query.get(query_name, [])
        assert len(kept) + len(dropped) <= requested_total
        no_mapping = requested_total - len(kept) - len(dropped)

        kept_mammals = sum(row["clade"] == "mammals" for row in kept)
        kept_non_mammals = len(kept) - kept_mammals
        recovered_clades = sorted(
            {str(row["clade"]) for row in kept},
            key=lambda clade: max(
                _as_int(row["phylogenetic_rank"])
                for row in kept
                if row["clade"] == clade
            ),
        )
        deepest = max(
            kept,
            key=lambda row: _as_int(row["phylogenetic_rank"]),
            default=None,
        )
        per_anchor_rows.append(
            {
                **{column: anchor[column] for column in required_anchors},
                "requested_mammal_species": requested_mammals,
                "requested_non_mammal_species": requested_non_mammals,
                "accepted_mammal_projections": kept_mammals,
                "accepted_non_mammal_projections": kept_non_mammals,
                "accepted_total_projections": len(kept),
                "requested_total_species": requested_total,
                "recovered_fraction": len(kept) / requested_total,
                "recovered_clades": ",".join(recovered_clades),
                "deepest_recovered_clade": (
                    "" if deepest is None else str(deepest["clade"])
                ),
                "no_mapping_count": no_mapping,
            }
        )

        existing_reason_counts: dict[str, int] = {}
        for row in dropped:
            reason = str(row["rejection_reason"])
            existing_reason_counts[reason] = existing_reason_counts.get(reason, 0) + 1
        if no_mapping:
            existing_reason_counts["no_mapping"] = no_mapping
        for reason, count in sorted(existing_reason_counts.items()):
            rejection_rows.append(
                {
                    "query_name": query_name,
                    "region_label": str(anchor["region_label"]),
                    "split": str(anchor["split"]),
                    "rejection_reason": reason,
                    "count": count,
                }
            )

        for scope in scopes:
            backend = scope["backend"]
            clade = scope["clade"]
            requested = _as_int(scope["requested_species"])
            recovered = sum(
                row["alignment_source"] == backend and row["clade"] == clade
                for row in kept
            )
            scope_rows.append(
                {
                    "query_name": query_name,
                    "region_label": str(anchor["region_label"]),
                    "split": str(anchor["split"]),
                    "backend": backend,
                    "clade": clade,
                    "requested_species": requested,
                    "recovered_species": recovered,
                    "recovered_fraction": recovered / requested,
                    "reached_clade": recovered > 0,
                }
            )

    per_anchor = pl.DataFrame(per_anchor_rows).sort("query_name")
    per_anchor_scope = pl.DataFrame(scope_rows).sort("query_name", "backend", "clade")
    rejection_schema = {
        "query_name": pl.String,
        "region_label": pl.String,
        "split": pl.String,
        "rejection_reason": pl.String,
        "count": pl.Int64,
    }
    rejection_counts = (
        pl.DataFrame(rejection_rows, schema=rejection_schema)
        if rejection_rows
        else pl.DataFrame(schema=rejection_schema)
    )
    aggregates = (
        per_anchor_scope.group_by("region_label", "split", "backend", "clade")
        .agg(
            n_anchors=pl.len(),
            mean_recovered_species=pl.col("recovered_species").mean(),
            median_recovered_species=pl.col("recovered_species").median(),
            q10_recovered_species=pl.col("recovered_species").quantile(0.10),
            q25_recovered_species=pl.col("recovered_species").quantile(0.25),
            q75_recovered_species=pl.col("recovered_species").quantile(0.75),
            q90_recovered_species=pl.col("recovered_species").quantile(0.90),
            mean_recovered_fraction=pl.col("recovered_fraction").mean(),
            fraction_anchors_reaching_clade=pl.col("reached_clade").mean(),
        )
        .sort("region_label", "split", "backend", "clade")
    )
    return ProjectionQcTables(
        per_anchor=per_anchor,
        per_anchor_scope=per_anchor_scope,
        rejection_counts=rejection_counts,
        aggregates=aggregates,
    )
