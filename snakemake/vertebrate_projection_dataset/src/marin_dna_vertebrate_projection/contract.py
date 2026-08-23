"""Center-projection acceptance and resizing contract."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from marin_dna_vertebrate_projection.maf import FRAGMENT_SCHEMA
from marin_dna_vertebrate_projection.projection.resize import resize_dataframe

ACCEPTED_SCHEMA = pl.Schema(
    {
        "query_name": pl.String,
        "source_chrom": pl.String,
        "source_start": pl.Int64,
        "source_end": pl.Int64,
        "region_label": pl.String,
        "species": pl.String,
        "alignment_name": pl.String,
        "assembly": pl.String,
        "taxonomy_id": pl.Int64,
        "family": pl.String,
        "clade": pl.String,
        "phylogenetic_rank": pl.Int64,
        "alignment_source": pl.String,
        "t_chrom": pl.String,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_strand": pl.String,
        "t_src_size": pl.Int64,
        "pre_resize_t_start": pl.Int64,
        "pre_resize_t_end": pl.Int64,
        "fragment_count": pl.Int64,
        "aligned_bases": pl.Int64,
    }
)

REJECTION_SCHEMA = pl.Schema(
    {
        "query_name": pl.String,
        "source_chrom": pl.String,
        "source_start": pl.Int64,
        "source_end": pl.Int64,
        "region_label": pl.String,
        "species": pl.String,
        "assembly": pl.String,
        "taxonomy_id": pl.Int64,
        "family": pl.String,
        "clade": pl.String,
        "phylogenetic_rank": pl.Int64,
        "alignment_source": pl.String,
        "rejection_reason": pl.String,
        "detail": pl.String,
        "fragment_count": pl.Int64,
    }
)

_METADATA_COLUMNS = [
    "source_chrom",
    "source_start",
    "source_end",
    "region_label",
    "alignment_name",
    "assembly",
    "taxonomy_id",
    "family",
    "clade",
    "phylogenetic_rank",
    "alignment_source",
]

TARGET_LENGTH = 255
PROJECTED_SPAN = (1, 2)


@dataclass(frozen=True)
class ProjectionContractResult:
    """Accepted projections and one explicit rejection row per rejected group."""

    accepted: pl.DataFrame
    rejected: pl.DataFrame


def apply_projection_contract(
    fragments: pl.DataFrame,
) -> ProjectionContractResult:
    """Apply the auditable contract with vectorized group summaries."""
    missing = set(FRAGMENT_SCHEMA) - set(fragments.columns)
    assert not missing, f"projection fragments missing columns: {sorted(missing)}"
    if fragments.is_empty():
        return ProjectionContractResult(
            accepted=pl.DataFrame(schema=ACCEPTED_SCHEMA),
            rejected=pl.DataFrame(schema=REJECTION_SCHEMA),
        )

    keys = ["query_name", "species"]
    target_ordered = fragments.sort(
        *keys, "t_start", "t_end", "fragment_id"
    ).with_columns(
        pl.col("t_end").cum_max().shift(1).over(keys).alias("_previous_target_end")
    )
    source_ordered = fragments.filter(
        pl.col("source_fragment_start").is_not_null()
        & pl.col("source_fragment_end").is_not_null()
    )
    if source_ordered.is_empty():
        source_overlaps = pl.DataFrame(
            schema={
                "query_name": pl.String,
                "species": pl.String,
                "_source_overlap": pl.Boolean,
            }
        )
    else:
        source_overlaps = (
            source_ordered.sort(
                *keys,
                "source_fragment_start",
                "source_fragment_end",
                "fragment_id",
            )
            .with_columns(
                pl.col("source_fragment_end")
                .cum_max()
                .shift(1)
                .over(keys)
                .alias("_previous_source_end")
            )
            .group_by(keys)
            .agg(
                (pl.col("source_fragment_start") < pl.col("_previous_source_end"))
                .fill_null(False)
                .any()
                .alias("_source_overlap")
            )
        )

    metadata_aggregations: list[pl.Expr] = []
    for column in _METADATA_COLUMNS:
        metadata_aggregations.extend(
            [
                pl.col(column).first().alias(column),
                pl.col(column).n_unique().alias(f"_n_{column}"),
            ]
        )
    summary = (
        target_ordered.group_by(keys)
        .agg(
            *metadata_aggregations,
            pl.col("t_chrom").first().alias("t_chrom"),
            pl.col("t_chrom").n_unique().alias("_n_t_chrom"),
            pl.col("t_chrom").unique().sort().alias("_t_chrom_values"),
            pl.col("t_strand").first().alias("t_strand"),
            pl.col("t_strand").n_unique().alias("_n_t_strand"),
            pl.col("t_strand").unique().sort().alias("_t_strand_values"),
            pl.col("t_src_size").first().alias("t_src_size"),
            pl.col("t_src_size").n_unique().alias("_n_t_src_size"),
            (
                pl.col("source_fragment_start").is_not_null()
                & pl.col("source_fragment_end").is_not_null()
                & (
                    (pl.col("source_fragment_start") < pl.col("source_start"))
                    | (pl.col("source_fragment_end") > pl.col("source_end"))
                    | (pl.col("source_fragment_end") <= pl.col("source_fragment_start"))
                )
            )
            .any()
            .alias("_invalid_source_fragment"),
            (
                (pl.col("t_start") < 0)
                | (pl.col("t_end") <= pl.col("t_start"))
                | (pl.col("t_end") > pl.col("t_src_size"))
            )
            .any()
            .alias("_invalid_target_bounds"),
            pl.col("fragment_id").n_unique().alias("_n_fragment_id"),
            (pl.col("t_start") < pl.col("_previous_target_end"))
            .fill_null(False)
            .any()
            .alias("_target_overlap"),
            pl.col("t_start").min().alias("pre_resize_t_start"),
            pl.col("t_end").max().alias("pre_resize_t_end"),
            pl.len().cast(pl.Int64).alias("fragment_count"),
            pl.col("aligned_bases").sum().alias("aligned_bases"),
        )
        .join(source_overlaps, on=keys, how="left", validate="1:1")
        .with_columns(pl.col("_source_overlap").fill_null(False))
    )

    metadata_flags = [pl.col(f"_n_{column}") != 1 for column in _METADATA_COLUMNS]
    inconsistent_metadata = pl.any_horizontal(metadata_flags)
    inconsistent_columns = pl.concat_str(
        [
            pl.when(flag).then(pl.lit(column)).otherwise(pl.lit(None, dtype=pl.String))
            for column, flag in zip(_METADATA_COLUMNS, metadata_flags, strict=True)
        ],
        separator=",",
        ignore_nulls=True,
    )
    pre_resize_length = pl.col("pre_resize_t_end") - pl.col("pre_resize_t_start")
    target_midpoint = (pl.col("pre_resize_t_start") + pl.col("pre_resize_t_end")) // 2
    centered_start = target_midpoint - TARGET_LENGTH // 2
    centered_end = centered_start + TARGET_LENGTH
    target_window_out_of_bounds = (centered_start < 0) | (
        centered_end > pl.col("t_src_size")
    )
    invalid_projected_span = ~pre_resize_length.is_between(*PROJECTED_SPAN)
    duplicated_fragment_id = pl.col("_n_fragment_id") != pl.col("fragment_count")
    rejection_reason = (
        pl.when(inconsistent_metadata)
        .then(pl.lit("inconsistent_metadata"))
        .when(pl.col("_n_t_chrom") != 1)
        .then(pl.lit("multi_chromosome"))
        .when(pl.col("_n_t_strand") != 1)
        .then(pl.lit("multi_strand"))
        .when(pl.col("_n_t_src_size") != 1)
        .then(pl.lit("inconsistent_target_size"))
        .when(pl.col("_invalid_source_fragment"))
        .then(pl.lit("invalid_source_fragment"))
        .when(pl.col("_invalid_target_bounds"))
        .then(pl.lit("invalid_target_bounds"))
        .when(duplicated_fragment_id | pl.col("_source_overlap"))
        .then(pl.lit("duplicated_mapping"))
        .when(pl.col("_target_overlap"))
        .then(pl.lit("ambiguous_mapping"))
        .when(invalid_projected_span)
        .then(pl.lit("invalid_projected_span"))
        .when(pl.col("t_src_size") < TARGET_LENGTH)
        .then(pl.lit("target_chromosome_too_short"))
        .when(target_window_out_of_bounds)
        .then(pl.lit("target_window_out_of_bounds"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    rejection_detail = (
        pl.when(inconsistent_metadata)
        .then(pl.concat_str(pl.lit("non-unique columns: "), inconsistent_columns))
        .when(pl.col("_n_t_chrom") != 1)
        .then(pl.col("_t_chrom_values").list.join(","))
        .when(pl.col("_n_t_strand") != 1)
        .then(pl.col("_t_strand_values").list.join(","))
        .when(pl.col("_n_t_src_size") != 1)
        .then(pl.lit("multiple src sizes"))
        .when(pl.col("_invalid_source_fragment"))
        .then(pl.lit("outside source anchor bounds"))
        .when(pl.col("_invalid_target_bounds"))
        .then(
            pl.concat_str(pl.lit("t_src_size="), pl.col("t_src_size").cast(pl.String))
        )
        .when(duplicated_fragment_id)
        .then(pl.lit("duplicate fragment IDs"))
        .when(pl.col("_source_overlap"))
        .then(pl.lit("overlapping source coverage"))
        .when(pl.col("_target_overlap"))
        .then(pl.lit("overlapping target fragments"))
        .when(invalid_projected_span)
        .then(
            pl.concat_str(
                pl.lit("length="),
                pre_resize_length.cast(pl.String),
                pl.lit(f"; expected={PROJECTED_SPAN[0]}..{PROJECTED_SPAN[1]}"),
            )
        )
        .when(pl.col("t_src_size") < TARGET_LENGTH)
        .then(
            pl.concat_str(
                pl.lit("t_src_size="),
                pl.col("t_src_size").cast(pl.String),
                pl.lit(f"; expected={TARGET_LENGTH}"),
            )
        )
        .when(target_window_out_of_bounds)
        .then(
            pl.concat_str(
                pl.lit("centered=["),
                centered_start.cast(pl.String),
                pl.lit(", "),
                centered_end.cast(pl.String),
                pl.lit("); t_src_size="),
                pl.col("t_src_size").cast(pl.String),
            )
        )
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    classified = summary.with_columns(
        rejection_reason.alias("_rejection_reason"),
        rejection_detail.alias("_rejection_detail"),
        pl.col("pre_resize_t_start").alias("t_start"),
        pl.col("pre_resize_t_end").alias("t_end"),
    )
    accepted = classified.filter(pl.col("_rejection_reason").is_null()).select(
        *ACCEPTED_SCHEMA.names()
    )
    accepted = resize_dataframe(accepted, TARGET_LENGTH).cast(ACCEPTED_SCHEMA)
    accepted_midpoints = (
        accepted["pre_resize_t_start"] + accepted["pre_resize_t_end"]
    ) // 2
    assert (accepted_midpoints - accepted["t_start"] == TARGET_LENGTH // 2).all()
    rejected = (
        classified.filter(pl.col("_rejection_reason").is_not_null())
        .select(
            *REJECTION_SCHEMA.names()[:-3],
            pl.col("_rejection_reason").alias("rejection_reason"),
            pl.col("_rejection_detail").alias("detail"),
            "fragment_count",
        )
        .cast(REJECTION_SCHEMA)
    )
    assert accepted.height + rejected.height == summary.height
    assert accepted.select(*keys).is_unique().all()
    assert rejected.select(*keys).is_unique().all()
    return ProjectionContractResult(
        accepted=accepted.sort(*keys), rejected=rejected.sort(*keys)
    )
