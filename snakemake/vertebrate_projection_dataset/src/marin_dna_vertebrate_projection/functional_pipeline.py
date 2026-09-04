"""File-level orchestration for the issue #517 functional-anchor workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


_DEVELOPMENT_SUBSETS = (
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "distal",
    "missense_variant",
    "non_coding_transcript_exon_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
)


def _read_development_variants(path: str | Path) -> tuple[pl.DataFrame, int]:
    """Read the pinned odd-autosome/X split and remove mature-miRNA groups."""
    required = {"chrom", "pos", "label", "subset", "match_group"}
    frame = pl.read_parquet(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"development VEP data missing columns: {sorted(missing)}")
    frame = frame.select(*sorted(required)).with_columns(
        pl.col("chrom").cast(pl.String),
        pl.col("pos").cast(pl.Int64),
        pl.col("label").cast(pl.Boolean),
        pl.col("subset").cast(pl.String),
        pl.col("match_group").cast(pl.Int64),
    )
    if (frame["pos"] <= 0).any():
        raise ValueError("development VEP positions must be 1-based positive")
    excluded_groups = frame.filter(pl.col("subset") == "mature_miRNA_variant")[
        "match_group"
    ].unique()
    frame = frame.filter(~pl.col("match_group").is_in(excluded_groups.implode()))
    allowed_chroms = {str(chrom) for chrom in range(1, 23, 2)} | {"X"}
    observed_chroms = set(frame["chrom"])
    if not observed_chroms <= allowed_chroms:
        raise AssertionError(
            "development VEP data contains held-out chromosomes: "
            f"{sorted(observed_chroms - allowed_chroms)}"
        )
    observed_subsets = set(frame["subset"])
    if observed_subsets != set(_DEVELOPMENT_SUBSETS):
        raise AssertionError(
            f"unexpected development VEP subsets: {sorted(observed_subsets)}"
        )
    return (
        frame.with_row_index("variant_id").with_columns(
            (pl.col("pos") - 1).alias("variant_start")
        ),
        excluded_groups.len(),
    )


def _development_anchor_pairs(
    anchors: pl.DataFrame, variants: pl.DataFrame
) -> pl.DataFrame:
    """Return bounded point-in-window pairs for 0-based half-open anchors."""
    required = {"query_name", "source_arm", "chrom", "start", "end"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"anchor catalog missing columns: {sorted(missing)}")
    if not (anchors["end"] - anchors["start"] == 255).all():
        raise AssertionError("development overlap requires 255 bp anchors")
    rows: list[dict[str, object]] = []
    for arm in FUNCTIONAL_ARMS:
        arm_anchors = anchors.filter(pl.col("source_arm") == arm)
        if arm_anchors.is_empty():
            raise AssertionError(f"development overlap catalog lacks arm {arm}")
        for chrom in variants["chrom"].unique().sort():
            chrom_anchors = arm_anchors.filter(pl.col("chrom") == chrom).sort(
                "start", "end", "query_name"
            )
            if chrom_anchors.is_empty():
                continue
            starts = chrom_anchors["start"].to_numpy()
            ends = chrom_anchors["end"].to_numpy()
            names = chrom_anchors["query_name"].to_list()
            chrom_variants = variants.filter(pl.col("chrom") == chrom)
            for variant_id, point in chrom_variants.select(
                "variant_id", "variant_start"
            ).iter_rows():
                first = int(np.searchsorted(starts, int(point) - 254, side="left"))
                last = int(np.searchsorted(starts, int(point), side="right"))
                for index in range(first, last):
                    if int(ends[index]) > int(point):
                        rows.append(
                            {
                                "variant_id": int(variant_id),
                                "query_name": names[index],
                                "source_arm": arm,
                            }
                        )
    schema = {
        "variant_id": pl.UInt32,
        "query_name": pl.String,
        "source_arm": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).unique()


def write_development_locus_overlap(
    projection_path: str | Path,
    training_path: str | Path,
    variants_path: str | Path,
    output_path: str | Path,
    *,
    dataset_repo: str,
    dataset_revision: str,
    dataset_split: str,
) -> None:
    """Report development-only VEP locus overlap for both conservation bands."""
    if dataset_split != "train":
        raise ValueError("preprojection overlap must use the development train split")
    variants, excluded_group_count = _read_development_variants(variants_path)
    totals = variants.group_by("subset").agg(
        pl.len().alias("variant_count"),
        pl.col("label").sum().cast(pl.Int64).alias("positive_variant_count"),
    )
    outputs: list[pl.DataFrame] = []
    overlap_columns = ["query_name", "source_arm", "chrom", "start", "end"]
    for catalog_name, path in [
        ("projection_ge_0.10", projection_path),
        ("training_ge_0.20", training_path),
    ]:
        pairs = _development_anchor_pairs(
            pl.read_parquet(path, columns=overlap_columns),
            variants,
        )
        pair_details = pairs.join(
            variants.select("variant_id", "subset", "label"),
            on="variant_id",
            how="left",
            validate="m:1",
        )
        overlap = pair_details.group_by("source_arm", "subset").agg(
            pl.col("variant_id").n_unique().alias("overlapping_variant_count"),
            pl.col("variant_id")
            .filter(pl.col("label"))
            .n_unique()
            .alias("overlapping_positive_variant_count"),
            pl.col("query_name").n_unique().alias("overlapping_anchor_count"),
            pl.len().alias("anchor_variant_pairs"),
        )
        grid = pl.DataFrame(
            [
                {"source_arm": arm, "subset": subset}
                for arm in FUNCTIONAL_ARMS
                for subset in _DEVELOPMENT_SUBSETS
            ]
        )
        outputs.append(
            grid.join(totals, on="subset", how="left", validate="m:1")
            .join(overlap, on=["source_arm", "subset"], how="left")
            .with_columns(
                pl.col(
                    "overlapping_variant_count",
                    "overlapping_positive_variant_count",
                    "overlapping_anchor_count",
                    "anchor_variant_pairs",
                ).fill_null(0),
                pl.lit(catalog_name).alias("catalog"),
            )
            .with_columns(
                (pl.col("overlapping_variant_count") / pl.col("variant_count")).alias(
                    "variant_overlap_fraction"
                ),
                (
                    pl.col("overlapping_positive_variant_count")
                    / pl.col("positive_variant_count")
                ).alias("positive_variant_overlap_fraction"),
                pl.lit(dataset_repo).alias("dataset_repo"),
                pl.lit(dataset_revision).alias("dataset_revision"),
                pl.lit(dataset_split).alias("dataset_split"),
                pl.lit("1-based VEP pos -> 0-based [pos-1,pos)").alias(
                    "coordinate_conversion"
                ),
                pl.lit(excluded_group_count).alias(
                    "excluded_mature_mirna_match_groups"
                ),
            )
        )
    result = pl.concat(outputs, how="vertical").sort("catalog", "source_arm", "subset")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path, separator="\t")


def write_anchor_distribution_summary(
    audited_path: str | Path,
    output_path: str | Path,
    *,
    training_min: float,
) -> None:
    """Write arm-wise preprojection composition and conservation quantiles."""
    audited = pl.read_parquet(audited_path)
    metrics = (
        "source_arm_owned_fraction",
        "union_functional_fraction",
        "exon_fraction",
        "repeat_masked_fraction",
        "gc_fraction",
        "ambiguous_base_fraction",
        "proportion_conserved",
        "contributing_feature_count",
    )
    required = {"source_arm", "chrom", "start", "end", *metrics}
    missing = required - set(audited.columns)
    if missing:
        raise ValueError(f"human anchor audit missing columns: {sorted(missing)}")
    if audited.select("chrom", "start", "end").unique().height != audited.height:
        raise AssertionError("retained catalog has exact-coordinate duplicates")
    if (audited.filter(pl.col("source_arm") == "enhancer")["exon_fraction"] > 0).any():
        raise AssertionError("retained enhancer anchor overlaps an annotated exon")
    rows: list[dict[str, object]] = []
    catalogs = [
        ("projection_ge_0.10", audited),
        (
            "training_ge_0.20",
            audited.filter(pl.col("proportion_conserved") >= training_min),
        ),
    ]
    for catalog_name, catalog in catalogs:
        for arm in FUNCTIONAL_ARMS:
            arm_rows = catalog.filter(pl.col("source_arm") == arm)
            if arm_rows.is_empty():
                raise AssertionError(f"human anchor audit lacks {catalog_name} {arm}")
            for metric in metrics:
                values = arm_rows[metric].cast(pl.Float64).to_numpy()
                quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
                rows.append(
                    {
                        "catalog": catalog_name,
                        "source_arm": arm,
                        "metric": metric,
                        "count": len(values),
                        "min": float(np.min(values)),
                        "q05": float(quantiles[0]),
                        "q25": float(quantiles[1]),
                        "median": float(quantiles[2]),
                        "q75": float(quantiles[3]),
                        "q95": float(quantiles[4]),
                        "max": float(np.max(values)),
                        "mean": float(np.mean(values)),
                    }
                )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).sort("catalog", "source_arm", "metric").write_csv(
        output_path, separator="\t"
    )


def write_chromosome_summary(
    projection_path: str | Path,
    training_path: str | Path,
    output_path: str | Path,
) -> None:
    """Write chromosome counts and within-arm fractions for both catalogs."""
    outputs = []
    for catalog_name, path in [
        ("projection_ge_0.10", projection_path),
        ("training_ge_0.20", training_path),
    ]:
        outputs.append(
            pl.read_parquet(path)
            .group_by("source_arm", "source_chrom")
            .len("anchor_count")
            .with_columns(
                (
                    pl.col("anchor_count")
                    / pl.col("anchor_count").sum().over("source_arm")
                ).alias("within_arm_fraction"),
                pl.lit(catalog_name).alias("catalog"),
            )
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pl.concat(outputs, how="vertical").sort(
        "catalog", "source_arm", "source_chrom"
    ).write_csv(output_path, separator="\t")


def write_drop_summaries(
    ownership_path: str | Path,
    construction_drops_path: str | Path,
    construction_output_path: str | Path,
    ownership_output_path: str | Path,
) -> None:
    """Summarize bounds/defined-sequence losses and ownership-gate losses."""
    ownership = pl.read_parquet(ownership_path)
    construction_drops = pl.read_parquet(construction_drops_path)
    construction_rows: list[dict[str, object]] = []
    ownership_rows: list[dict[str, object]] = []
    for arm in FUNCTIONAL_ARMS:
        arm_ownership = ownership.filter(pl.col("source_arm") == arm)
        arm_drops = construction_drops.filter(pl.col("source_arm") == arm)
        source_total = arm_ownership.height + arm_drops.height
        if source_total == 0:
            raise AssertionError(f"construction audit lacks arm {arm}")
        construction_rows.append(
            {
                "source_arm": arm,
                "outcome": "construction_valid",
                "window_count": arm_ownership.height,
                "source_total": source_total,
                "fraction": arm_ownership.height / source_total,
            }
        )
        for drop_reason, count in arm_drops.group_by("drop_reason").len().iter_rows():
            construction_rows.append(
                {
                    "source_arm": arm,
                    "outcome": str(drop_reason),
                    "window_count": int(count),
                    "source_total": source_total,
                    "fraction": int(count) / source_total,
                }
            )
        ownership_total = arm_ownership.height
        for passes, winner, count in (
            arm_ownership.group_by("passes_ownership_gate", "ownership_winner")
            .len()
            .iter_rows()
        ):
            ownership_rows.append(
                {
                    "source_arm": arm,
                    "outcome": "retained" if passes else f"lost_to_{winner}",
                    "window_count": int(count),
                    "source_total": ownership_total,
                    "fraction": int(count) / ownership_total,
                }
            )
    for path in [construction_output_path, ownership_output_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(construction_rows).sort("source_arm", "outcome").write_csv(
        construction_output_path, separator="\t"
    )
    pl.DataFrame(ownership_rows).sort("source_arm", "outcome").write_csv(
        ownership_output_path, separator="\t"
    )


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
