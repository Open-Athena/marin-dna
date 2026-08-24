"""File-level orchestration for the issue #517 functional-anchor workflow."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from marin_dna.data.intervals import GenomicSet
from marin_dna.data.utils import load_annotation
from marin_dna_vertebrate_projection.conservation.scoring import score_windows
from marin_dna_vertebrate_projection.functional_anchors import (
    DEFAULT_NCRNA_BIOTYPES,
    FUNCTIONAL_ARMS,
    annotate_sequence_fractions,
    apply_window_ownership_gate,
    build_candidate_windows,
    extract_functional_features,
    feature_audit_table,
    pairwise_raw_overlap_table,
    resolve_base_priority,
    split_conservation_catalogs,
    to_projection_catalog,
)


def read_ccre_v4(path: str | Path) -> pl.DataFrame:
    """Read the pinned ENCODE SCREEN Registry V4 BED used by issue #517."""
    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        columns=[0, 1, 2, 3, 5],
        new_columns=["chrom", "start", "end", "ccre_id", "cre_class"],
        schema_overrides={"chrom": pl.String},
    )
    if frame.is_empty():
        raise ValueError("cCRE V4 input is empty")
    if frame["ccre_id"].null_count() or frame["ccre_id"].n_unique() != frame.height:
        raise ValueError("cCRE V4 identifiers must be present and unique")
    if (frame["end"] <= frame["start"]).any():
        raise ValueError("cCRE V4 contains invalid 0-based half-open intervals")
    return frame


def read_chrom_sizes(path: str | Path) -> pl.DataFrame:
    """Read a two-column UCSC chromosome-size file."""
    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "size"],
        schema_overrides={"chrom": pl.String, "size": pl.Int64},
    )
    if frame["chrom"].n_unique() != frame.height or (frame["size"] <= 0).any():
        raise ValueError("chromosome sizes must be positive and unique")
    return frame


def build_functional_anchor_artifacts(
    gtf_path: str | Path,
    ccre_path: str | Path,
    chrom_sizes_path: str | Path,
    defined_bed_path: str | Path,
    *,
    retained_path: str | Path,
    provenance_path: str | Path,
    construction_drops_path: str | Path,
    ownership_audit_path: str | Path,
    feature_audit_path: str | Path,
    overlap_audit_path: str | Path,
    summary_path: str | Path,
    standard_chroms: list[str],
    ncrna_biotypes: list[str] | tuple[str, ...] = DEFAULT_NCRNA_BIOTYPES,
    priority: list[str] | tuple[str, ...] = FUNCTIONAL_ARMS,
    tss_radius: int = 256,
    window_size: int = 255,
    step_size: int = 128,
    feature_flank: int = 20,
    min_feature_size: int = 20,
    max_feature_size: int = 10_000,
) -> None:
    """Build and write all pre-conservation functional-anchor audit artifacts."""
    annotation = load_annotation(str(gtf_path))
    features = extract_functional_features(
        annotation,
        read_ccre_v4(ccre_path),
        standard_chroms=standard_chroms,
        ncrna_biotypes=ncrna_biotypes,
        tss_radius=tss_radius,
    )
    ownership = resolve_base_priority(features.raw_cores, priority=priority)
    candidates = build_candidate_windows(
        features,
        ownership,
        chrom_sizes=read_chrom_sizes(chrom_sizes_path),
        defined=GenomicSet.read_bed(str(defined_bed_path)),
        window_size=window_size,
        step_size=step_size,
        feature_flank=feature_flank,
        min_feature_size=min_feature_size,
        max_feature_size=max_feature_size,
    )
    gate = apply_window_ownership_gate(candidates.windows, features, ownership)
    paths = [
        retained_path,
        provenance_path,
        construction_drops_path,
        ownership_audit_path,
        feature_audit_path,
        overlap_audit_path,
        summary_path,
    ]
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    gate.retained.write_parquet(retained_path)
    candidates.provenance.write_parquet(provenance_path)
    candidates.construction_drops.write_parquet(construction_drops_path)
    gate.audit.write_parquet(ownership_audit_path)
    feature_audit_table(features, ownership).write_csv(
        feature_audit_path, separator="\t"
    )
    pairwise_raw_overlap_table(features.raw_cores).write_csv(
        overlap_audit_path, separator="\t"
    )
    arm_counts = {
        arm: gate.retained.filter(pl.col("source_arm") == arm).height
        for arm in FUNCTIONAL_ARMS
    }
    if any(count == 0 for count in arm_counts.values()):
        raise AssertionError(f"construction retained an empty arm: {arm_counts}")
    Path(summary_path).write_text(
        json.dumps(
            {
                "annotation": "Ensembl GRCh38",
                "annotation_transcript_scope": "all_qualifying_transcripts",
                "candidate_counts": arm_counts,
                "coordinate_system": "0-based half-open",
                "enhancer_annotation": "ENCODE SCREEN Registry V4 dELS/pELS",
                "priority": list(priority),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def score_functional_candidates(
    candidates_path: str | Path,
    bigwig_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float,
) -> None:
    """Score retained candidates with the pinned phyloP base threshold."""
    candidates = pl.read_parquet(candidates_path)
    scored = score_windows(bigwig_path, candidates, threshold)
    if scored.height != candidates.height:
        raise AssertionError("phyloP scoring changed candidate row count")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(output_path)


def _deterministic_head(frame: pl.DataFrame, count: int, seed: int) -> pl.DataFrame:
    if count <= 0 or frame.height <= count:
        return frame
    return (
        frame.with_columns(pl.col("query_name").hash(seed=seed).alias("_sample_hash"))
        .sort("_sample_hash", "query_name")
        .head(count)
        .drop("_sample_hash")
    )


def write_conservation_catalogs(
    scored_path: str | Path | list[str],
    projection_path: str | Path,
    training_path: str | Path,
    deferred_path: str | Path,
    summary_path: str | Path,
    *,
    projection_min: float,
    training_min: float,
    smoke_training_per_arm: int | None = None,
    smoke_deferred_per_arm: int | None = None,
    seed: int = 517,
) -> None:
    """Write nested projection/training catalogs, with optional smoke caps."""
    scored = (
        pl.concat([pl.read_parquet(path) for path in scored_path], how="vertical")
        if isinstance(scored_path, list)
        else pl.read_parquet(scored_path)
    )
    catalogs = split_conservation_catalogs(
        scored,
        projection_min=projection_min,
        training_min=training_min,
    )
    training = catalogs.training
    deferred = catalogs.deferred
    if smoke_training_per_arm is not None:
        training = pl.concat(
            [
                _deterministic_head(
                    training.filter(pl.col("source_arm") == arm),
                    smoke_training_per_arm,
                    seed,
                )
                for arm in FUNCTIONAL_ARMS
            ],
            how="vertical",
        )
    if smoke_deferred_per_arm is not None:
        deferred = pl.concat(
            [
                _deterministic_head(
                    deferred.filter(pl.col("source_arm") == arm),
                    smoke_deferred_per_arm,
                    seed,
                )
                for arm in FUNCTIONAL_ARMS
            ],
            how="vertical",
        )
    projection = pl.concat([training, deferred], how="vertical").unique(
        subset="query_name"
    )
    counts: dict[str, dict[str, int]] = {}
    for arm in FUNCTIONAL_ARMS:
        counts[arm] = {
            "projection": projection.filter(pl.col("source_arm") == arm).height,
            "training": training.filter(pl.col("source_arm") == arm).height,
            "deferred": deferred.filter(pl.col("source_arm") == arm).height,
        }
    if any(values["training"] == 0 for values in counts.values()):
        raise AssertionError(
            f"conservation filter produced an empty training arm: {counts}"
        )
    if smoke_deferred_per_arm and any(
        values["deferred"] == 0 for values in counts.values()
    ):
        raise AssertionError(f"smoke catalog lacks a deferred example: {counts}")
    projection = to_projection_catalog(projection)
    training = to_projection_catalog(training)
    deferred = to_projection_catalog(deferred)
    for path in [projection_path, training_path, deferred_path, summary_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    projection.write_parquet(projection_path)
    training.write_parquet(training_path)
    deferred.write_parquet(deferred_path)
    Path(summary_path).write_text(
        json.dumps(
            {
                "counts": counts,
                "projection_min": projection_min,
                "training_min": training_min,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_human_anchor_audit(
    projection_catalog_path: str | Path,
    human_sequences_path: str | Path,
    output_path: str | Path,
) -> None:
    """Attach human sequence composition to the complete projection catalog."""
    catalog = pl.read_parquet(projection_catalog_path)
    sequences = pl.read_parquet(human_sequences_path).select("query_name", "sequence")
    audited = annotate_sequence_fractions(catalog, sequences)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    audited.write_parquet(output_path)


def write_training_sequences(
    combined_path: str | Path,
    training_catalog_path: str | Path,
    output_path: str | Path,
) -> None:
    """Keep only >= training-threshold anchors after the shared projection."""
    training_names = pl.scan_parquet(training_catalog_path).select("query_name")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(combined_path).join(
        training_names, on="query_name", how="semi"
    ).sink_parquet(output)
