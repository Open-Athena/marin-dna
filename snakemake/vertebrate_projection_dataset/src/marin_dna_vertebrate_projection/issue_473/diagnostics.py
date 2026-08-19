"""Paired recovery, locus, sequence, and anchor-level diagnostics for #473."""

from __future__ import annotations

from pathlib import Path

import polars as pl

PAIR_KEYS = ("query_name", "species")
_METADATA_COLUMNS = (
    "source_chrom",
    "source_start",
    "source_end",
    "region_label",
    "alignment_source",
    "assembly",
    "taxonomy_id",
    "family",
    "clade",
    "phylogenetic_rank",
)
_POLICY_COLUMNS = (
    "t_chrom",
    "t_start",
    "t_end",
    "t_strand",
    "pre_resize_t_start",
    "pre_resize_t_end",
    "fragment_count",
    "aligned_bases",
    "sequence",
)


def _accepted(frame: pl.LazyFrame, prefix: str) -> pl.LazyFrame:
    required = {*PAIR_KEYS, *_METADATA_COLUMNS, *_POLICY_COLUMNS}
    missing = required - set(frame.collect_schema().names())
    assert not missing, f"{prefix} accepted rows missing columns: {sorted(missing)}"
    return frame.select(
        *PAIR_KEYS,
        *[
            pl.col(column).alias(f"{prefix}_{column}")
            for column in (*_METADATA_COLUMNS, *_POLICY_COLUMNS)
        ],
        pl.lit(True).alias(f"{prefix}_accepted"),
    )


def _rejections(frame: pl.LazyFrame, prefix: str) -> pl.LazyFrame:
    required = {*PAIR_KEYS, "rejection_reason", "detail"}
    missing = required - set(frame.collect_schema().names())
    assert not missing, f"{prefix} rejections missing columns: {sorted(missing)}"
    return frame.group_by(*PAIR_KEYS).agg(
        pl.col("rejection_reason").first().alias(f"{prefix}_rejection_reason"),
        pl.col("detail").first().alias(f"{prefix}_rejection_detail"),
        pl.len().alias(f"{prefix}_rejection_rows"),
    )


def _sequence_metrics(prefix: str) -> list[pl.Expr]:
    sequence = pl.col(f"{prefix}_sequence")
    sequence_length = sequence.str.len_chars()
    lowercase = sequence.str.count_matches("[a-z]")
    canonical = sequence.str.to_lowercase().str.count_matches("[acgt]")
    gc = sequence.str.to_lowercase().str.count_matches("[gc]")
    return [
        (gc / sequence_length).alias(f"{prefix}_gc_fraction"),
        (1.0 - canonical / sequence_length).alias(f"{prefix}_ambiguous_base_fraction"),
        (lowercase / sequence_length).alias(f"{prefix}_repeat_masked_fraction"),
    ]


def build_paired_union(
    full_window: pl.LazyFrame,
    center_1: pl.LazyFrame,
    full_window_rejections: pl.LazyFrame,
    center_1_rejections: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build one row per accepted-union pair with explicit missing-side status."""
    full = _accepted(full_window, "full_window")
    center = _accepted(center_1, "center_1")
    paired = (
        full.join(center, on=list(PAIR_KEYS), how="full", coalesce=True)
        .with_columns(
            pl.col("full_window_accepted").fill_null(False),
            pl.col("center_1_accepted").fill_null(False),
        )
        .join(
            _rejections(full_window_rejections, "full_window"),
            on=list(PAIR_KEYS),
            how="left",
        )
        .join(
            _rejections(center_1_rejections, "center_1"),
            on=list(PAIR_KEYS),
            how="left",
        )
        .with_columns(
            *[
                pl.coalesce(
                    pl.col(f"full_window_{column}"),
                    pl.col(f"center_1_{column}"),
                ).alias(column)
                for column in _METADATA_COLUMNS
            ],
            pl.when(pl.col("full_window_accepted") & pl.col("center_1_accepted"))
            .then(pl.lit("both"))
            .when(pl.col("full_window_accepted"))
            .then(pl.lit("full_window_only"))
            .otherwise(pl.lit("center_1_only"))
            .alias("pair_outcome"),
            pl.when(pl.col("full_window_accepted"))
            .then(pl.lit("accepted"))
            .when(pl.col("full_window_rejection_reason").is_not_null())
            .then(pl.lit("rejected:") + pl.col("full_window_rejection_reason"))
            .otherwise(pl.lit("no_mapping"))
            .alias("full_window_status"),
            pl.when(pl.col("center_1_accepted"))
            .then(pl.lit("accepted"))
            .when(pl.col("center_1_rejection_reason").is_not_null())
            .then(pl.lit("rejected:") + pl.col("center_1_rejection_reason"))
            .otherwise(pl.lit("no_mapping"))
            .alias("center_1_status"),
        )
    )
    both = pl.col("pair_outcome") == "both"
    same_chrom = pl.col("full_window_t_chrom") == pl.col("center_1_t_chrom")
    same_strand = pl.col("full_window_t_strand") == pl.col("center_1_t_strand")
    locus_overlap = same_chrom & (
        pl.max_horizontal(
            pl.col("full_window_t_start"),
            pl.col("center_1_t_start"),
        )
        < pl.min_horizontal(
            pl.col("full_window_t_end"),
            pl.col("center_1_t_end"),
        )
    )
    exact_locus = (
        same_chrom
        & (pl.col("full_window_t_start") == pl.col("center_1_t_start"))
        & (pl.col("full_window_t_end") == pl.col("center_1_t_end"))
    )
    full_center = (pl.col("full_window_t_start") + pl.col("full_window_t_end")) // 2
    center_center = (pl.col("center_1_t_start") + pl.col("center_1_t_end")) // 2
    center_pre_start = pl.col("center_1_pre_resize_t_start")
    center_pre_end = pl.col("center_1_pre_resize_t_end")
    center_left_genomic = center_pre_start - pl.col("center_1_t_start")
    center_right_genomic = pl.col("center_1_t_end") - center_pre_end
    return paired.with_columns(
        pl.when(both).then(same_chrom).alias("target_chrom_agreement"),
        pl.when(both).then(same_strand).alias("target_strand_agreement"),
        pl.when(both).then(locus_overlap).alias("target_locus_overlap"),
        pl.when(both).then(exact_locus).alias("target_locus_exact"),
        pl.when(both & same_chrom)
        .then((full_center - center_center).abs())
        .alias("emitted_center_displacement_bases"),
        (pl.col("full_window_aligned_bases") / pl.lit(255.0)).alias(
            "full_window_landmark_aligned_fraction"
        ),
        pl.col("center_1_aligned_bases")
        .cast(pl.Float64)
        .alias("center_1_landmark_aligned_fraction"),
        pl.when(pl.col("center_1_t_strand") == "+")
        .then(center_left_genomic)
        .otherwise(center_right_genomic)
        .alias("center_1_human_oriented_left_flank_bases"),
        pl.when(pl.col("center_1_t_strand") == "+")
        .then(center_right_genomic)
        .otherwise(center_left_genomic)
        .alias("center_1_human_oriented_right_flank_bases"),
        pl.lit(None, dtype=pl.Float64).alias("emitted_window_aligned_coverage"),
        pl.lit("unavailable_genome_wide").alias(
            "emitted_window_aligned_coverage_status"
        ),
        pl.lit("target_span_geometry_not_alignment_coverage").alias(
            "flank_metric_status"
        ),
        *_sequence_metrics("full_window"),
        *_sequence_metrics("center_1"),
    )


def _scope_summary(
    rows: pl.LazyFrame, scope_type: str, column: str | None
) -> pl.LazyFrame:
    scoped = rows.with_columns(
        pl.lit(scope_type).alias("scope_type"),
        (
            pl.lit("all")
            if column is None
            else pl.col(column).cast(pl.String).fill_null("unknown")
        ).alias("scope_value"),
    )
    return scoped.group_by("scope_type", "scope_value").agg(
        pl.len().cast(pl.Int64).alias("accepted_union"),
        (pl.col("pair_outcome") == "both")
        .sum()
        .cast(pl.Int64)
        .alias("accepted_by_both"),
        (pl.col("pair_outcome") == "full_window_only")
        .sum()
        .cast(pl.Int64)
        .alias("full_window_only"),
        (pl.col("pair_outcome") == "center_1_only")
        .sum()
        .cast(pl.Int64)
        .alias("center_1_only"),
        pl.col("target_chrom_agreement")
        .cast(pl.Float64)
        .mean()
        .alias("target_chrom_agreement_rate"),
        pl.col("target_strand_agreement")
        .cast(pl.Float64)
        .mean()
        .alias("target_strand_agreement_rate"),
        pl.col("target_locus_overlap")
        .cast(pl.Float64)
        .mean()
        .alias("target_locus_overlap_rate"),
        pl.col("target_locus_exact")
        .cast(pl.Float64)
        .mean()
        .alias("target_locus_exact_rate"),
        pl.col("emitted_center_displacement_bases")
        .median()
        .alias("median_emitted_center_displacement_bases"),
        pl.col("emitted_center_displacement_bases")
        .quantile(0.9)
        .alias("q90_emitted_center_displacement_bases"),
        pl.col("full_window_landmark_aligned_fraction")
        .mean()
        .alias("mean_full_window_landmark_aligned_fraction"),
        pl.col("center_1_landmark_aligned_fraction")
        .mean()
        .alias("mean_center_1_landmark_aligned_fraction"),
        pl.col("center_1_human_oriented_left_flank_bases")
        .mean()
        .alias("mean_center_1_left_flank_bases"),
        pl.col("center_1_human_oriented_right_flank_bases")
        .mean()
        .alias("mean_center_1_right_flank_bases"),
        pl.col("full_window_gc_fraction").mean().alias("mean_full_window_gc_fraction"),
        pl.col("center_1_gc_fraction").mean().alias("mean_center_1_gc_fraction"),
        pl.col("full_window_ambiguous_base_fraction")
        .mean()
        .alias("mean_full_window_ambiguous_base_fraction"),
        pl.col("center_1_ambiguous_base_fraction")
        .mean()
        .alias("mean_center_1_ambiguous_base_fraction"),
        pl.col("full_window_repeat_masked_fraction")
        .mean()
        .alias("mean_full_window_repeat_masked_fraction"),
        pl.col("center_1_repeat_masked_fraction")
        .mean()
        .alias("mean_center_1_repeat_masked_fraction"),
    )


def write_paired_diagnostics(
    full_window_path: str | Path,
    center_1_path: str | Path,
    full_window_rejection_paths: list[str],
    center_1_rejection_paths: list[str],
    anchor_catalog_path: str | Path,
    paired_path: str | Path,
    scope_summary_path: str | Path,
    per_anchor_path: str | Path,
    anchor_uncertainty_path: str | Path,
) -> None:
    """Write detailed accepted-union diagnostics and anchor-clustered uncertainty."""
    assert full_window_rejection_paths and center_1_rejection_paths
    full_rejected = pl.concat(
        [pl.scan_parquet(path) for path in full_window_rejection_paths],
        how="vertical",
    )
    center_rejected = pl.concat(
        [pl.scan_parquet(path) for path in center_1_rejection_paths],
        how="vertical",
    )
    paired = build_paired_union(
        pl.scan_parquet(full_window_path),
        pl.scan_parquet(center_1_path),
        full_rejected,
        center_rejected,
    )
    for path in [
        paired_path,
        scope_summary_path,
        per_anchor_path,
        anchor_uncertainty_path,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    paired.sink_parquet(paired_path)

    projection_rows = pl.scan_parquet(paired_path).filter(
        pl.col("alignment_source") != "human_reference"
    )
    pl.concat(
        [
            _scope_summary(projection_rows, "all", None),
            _scope_summary(projection_rows, "region_label", "region_label"),
            _scope_summary(projection_rows, "alignment_source", "alignment_source"),
            _scope_summary(projection_rows, "species", "species"),
            _scope_summary(projection_rows, "clade", "clade"),
        ],
        how="vertical",
    ).sort("scope_type", "scope_value").sink_parquet(scope_summary_path)

    counts = projection_rows.group_by("query_name").agg(
        pl.col("full_window_accepted")
        .sum()
        .cast(pl.Int64)
        .alias("full_window_accepted_species"),
        pl.col("center_1_accepted")
        .sum()
        .cast(pl.Int64)
        .alias("center_1_accepted_species"),
    )
    anchors = pl.scan_parquet(anchor_catalog_path).select(
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
    )
    per_anchor = (
        anchors.join(counts, on="query_name", how="left")
        .with_columns(
            pl.col("full_window_accepted_species").fill_null(0),
            pl.col("center_1_accepted_species").fill_null(0),
        )
        .with_columns(
            (
                pl.col("center_1_accepted_species")
                - pl.col("full_window_accepted_species")
            ).alias("accepted_species_delta")
        )
        .sort("region_label", "source_chrom", "source_start", "query_name")
    )
    per_anchor.sink_parquet(per_anchor_path)
    uncertainty = (
        pl.scan_parquet(per_anchor_path)
        .group_by("region_label")
        .agg(
            pl.len().cast(pl.Int64).alias("n_anchors"),
            pl.col("full_window_accepted_species")
            .mean()
            .alias("mean_full_window_accepted_species"),
            pl.col("center_1_accepted_species")
            .mean()
            .alias("mean_center_1_accepted_species"),
            pl.col("accepted_species_delta").mean().alias("mean_paired_delta"),
            pl.col("accepted_species_delta")
            .std()
            .fill_null(0.0)
            .alias("sd_paired_delta"),
            pl.col("accepted_species_delta").median().alias("median_paired_delta"),
            pl.col("accepted_species_delta").quantile(0.1).alias("q10_paired_delta"),
            pl.col("accepted_species_delta").quantile(0.9).alias("q90_paired_delta"),
        )
        .with_columns(
            (
                pl.col("sd_paired_delta") / pl.col("n_anchors").cast(pl.Float64).sqrt()
            ).alias("se_paired_delta")
        )
        .with_columns(
            (pl.col("mean_paired_delta") - 1.96 * pl.col("se_paired_delta")).alias(
                "normal_95ci_low"
            ),
            (pl.col("mean_paired_delta") + 1.96 * pl.col("se_paired_delta")).alias(
                "normal_95ci_high"
            ),
            pl.lit("anchor-clustered normal interval").alias("uncertainty_method"),
        )
        .sort("region_label")
    )
    uncertainty.sink_parquet(anchor_uncertainty_path)


def write_manual_pair_sample(
    paired_path: str | Path,
    sample_path: str | Path,
    report_path: str | Path,
    *,
    seed: int = 473,
    rows_per_category: int = 3,
) -> None:
    """Select bounded representative pairs, including the ZRS source locus."""
    assert rows_per_category > 0
    rows = pl.scan_parquet(paired_path)
    identity_hash = pl.concat_str(
        [
            pl.col("query_name"),
            pl.col("species"),
            pl.col("pair_outcome"),
        ],
        separator="\t",
    ).hash(seed=seed)
    categories: list[tuple[str, pl.Expr]] = [
        ("cds", pl.col("region_label") == "cds"),
        (
            "enhancer",
            pl.col("region_label") == "ccre_enhancer_centered",
        ),
        ("full_window_only", pl.col("pair_outcome") == "full_window_only"),
        ("center_1_only", pl.col("pair_outcome") == "center_1_only"),
        (
            "locus_disagreement",
            (pl.col("pair_outcome") == "both")
            & ~pl.col("target_locus_exact").fill_null(False),
        ),
        (
            "fragmented",
            (pl.col("full_window_fragment_count").fill_null(0) > 1)
            | (pl.col("center_1_fragment_count").fill_null(0) > 1),
        ),
        (
            "zrs",
            (pl.col("source_chrom") == "chr7")
            & (pl.col("source_start") < 156_793_500)
            & (pl.col("source_end") > 156_791_000),
        ),
    ]
    samples: list[pl.DataFrame] = []
    for category, condition in categories:
        selected = (
            rows.filter(condition)
            .with_columns(
                pl.lit(category).alias("inspection_category"),
                identity_hash.alias("_sample_hash"),
            )
            .bottom_k(rows_per_category, by="_sample_hash")
            .drop("_sample_hash")
            .collect(engine="streaming")
        )
        samples.append(selected)
    sample = pl.concat(samples, how="vertical").unique(
        subset=["inspection_category", *PAIR_KEYS], maintain_order=True
    )
    assert sample.height > 0
    sample_output = Path(sample_path)
    report_output = Path(report_path)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(sample_output, separator="\t")

    display_columns = [
        "inspection_category",
        "query_name",
        "species",
        "region_label",
        "pair_outcome",
        "full_window_status",
        "center_1_status",
        "full_window_t_chrom",
        "full_window_t_start",
        "full_window_t_end",
        "full_window_t_strand",
        "center_1_t_chrom",
        "center_1_t_start",
        "center_1_t_end",
        "center_1_t_strand",
        "emitted_center_displacement_bases",
    ]
    lines = [
        "# Issue #473 paired manual inspection",
        "",
        f"Deterministic seed: {seed}. Coordinates are 0-based and half-open.",
        "",
        (
            "Exact emitted-window alignment coverage is not inferred from target span; "
            "the genome-wide field is explicitly unavailable pending sampled traces."
        ),
        "",
        "| " + " | ".join(display_columns) + " |",
        "|" + "|".join(["---"] * len(display_columns)) + "|",
    ]
    for row in sample.select(display_columns).iter_rows(named=True):
        lines.append(
            "| "
            + " | ".join(
                "" if row[column] is None else str(row[column])
                for column in display_columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "- [ ] Inspect human and target coordinates against the raw HAL/MAF trace.",
            "- [ ] Confirm target strand and human-oriented sequence.",
            "- [ ] Review center-only and full-window-only rejection/no-mapping status.",
            "- [ ] Confirm at least one ZRS row when the fixed enhancer population overlaps it.",
        ]
    )
    report_output.write_text("\n".join(lines) + "\n")
