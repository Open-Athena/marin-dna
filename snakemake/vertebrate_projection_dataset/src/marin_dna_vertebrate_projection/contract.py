"""Shared projection acceptance, resizing, and sequence-extraction contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from marin_dna_vertebrate_projection.maf import FRAGMENT_SCHEMA
from marin_dna_vertebrate_projection.projection.resize import (
    resize_dataframe,
    resize_to_length,
)

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

SequenceFetcher = Callable[[str, str, int, int], str | None]


@dataclass(frozen=True)
class ProjectionContractResult:
    """Accepted projections and one explicit rejection row per rejected group."""

    accepted: pl.DataFrame
    rejected: pl.DataFrame


@dataclass(frozen=True)
class SequenceExtractionResult:
    """Sequence-bearing accepted rows and extraction-time rejections."""

    accepted: pl.DataFrame
    rejected: pl.DataFrame


def _has_overlapping_intervals(intervals: list[tuple[int, int]]) -> bool:
    if len(intervals) < 2:
        return False
    sorted_intervals = sorted(intervals)
    max_end = sorted_intervals[0][1]
    for start, end in sorted_intervals[1:]:
        if start < max_end:
            return True
        max_end = max(max_end, end)
    return False


def _group_value(group: pl.DataFrame, column: str) -> object:
    values = group[column].drop_nulls().unique().to_list()
    assert values, f"projection group has no value for {column}"
    return values[0]


def _as_int(value: object) -> int:
    assert isinstance(value, int | str) and not isinstance(value, bool)
    return int(value)


def _rejection(group: pl.DataFrame, reason: str, detail: str) -> dict[str, object]:
    return {
        "query_name": str(_group_value(group, "query_name")),
        "source_chrom": str(_group_value(group, "source_chrom")),
        "source_start": _as_int(_group_value(group, "source_start")),
        "source_end": _as_int(_group_value(group, "source_end")),
        "region_label": str(_group_value(group, "region_label")),
        "species": str(_group_value(group, "species")),
        "assembly": str(_group_value(group, "assembly")),
        "taxonomy_id": _as_int(_group_value(group, "taxonomy_id")),
        "family": str(_group_value(group, "family")),
        "clade": str(_group_value(group, "clade")),
        "phylogenetic_rank": _as_int(_group_value(group, "phylogenetic_rank")),
        "alignment_source": str(_group_value(group, "alignment_source")),
        "rejection_reason": reason,
        "detail": detail,
        "fragment_count": group.height,
    }


def _apply_fragmented_projection_contract(
    fragments: pl.DataFrame,
    *,
    target_length: int = 255,
    pre_resize_min_length: int = 1,
    pre_resize_max_length: int = 2,
) -> ProjectionContractResult:
    """Apply the detailed contract to groups containing multiple fragments.

    Coordinates entering this function must already be 0-based and half-open.
    Fragmented mappings are accepted only when every fragment agrees on target
    chromosome, strand, metadata, and bounds.  Overlapping source or target
    fragments are treated as duplicated/ambiguous mappings.
    """
    missing = set(FRAGMENT_SCHEMA) - set(fragments.columns)
    assert not missing, f"projection fragments missing columns: {sorted(missing)}"
    assert target_length > 0
    group_sizes = fragments.group_by("query_name", "species").len()["len"]
    assert group_sizes.is_empty() or (group_sizes > 1).all(), (
        "fragmented contract received a single-fragment group"
    )
    assert 0 < pre_resize_min_length <= pre_resize_max_length

    if fragments.is_empty():
        return ProjectionContractResult(
            accepted=pl.DataFrame(schema=ACCEPTED_SCHEMA),
            rejected=pl.DataFrame(schema=REJECTION_SCHEMA),
        )

    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    groups = fragments.partition_by(
        ["query_name", "species"], as_dict=True, maintain_order=True
    )
    for group in groups.values():
        inconsistent = [
            column for column in _METADATA_COLUMNS if group[column].n_unique() != 1
        ]
        if inconsistent:
            rejected_rows.append(
                _rejection(
                    group,
                    "inconsistent_metadata",
                    f"non-unique columns: {','.join(inconsistent)}",
                )
            )
            continue
        if group["t_chrom"].n_unique() != 1:
            rejected_rows.append(
                _rejection(
                    group,
                    "multi_chromosome",
                    ",".join(sorted(group["t_chrom"].unique().to_list())),
                )
            )
            continue
        if group["t_strand"].n_unique() != 1:
            rejected_rows.append(
                _rejection(
                    group,
                    "multi_strand",
                    ",".join(sorted(group["t_strand"].unique().to_list())),
                )
            )
            continue
        if group["t_src_size"].n_unique() != 1:
            rejected_rows.append(
                _rejection(group, "inconsistent_target_size", "multiple src sizes")
            )
            continue

        source_start = _as_int(_group_value(group, "source_start"))
        source_end = _as_int(_group_value(group, "source_end"))
        source_fragment_rows = group.select(
            "source_fragment_start", "source_fragment_end"
        ).drop_nulls()
        source_fragments = [
            (int(start), int(end)) for start, end in source_fragment_rows.iter_rows()
        ]
        if any(
            start < source_start or end > source_end or end <= start
            for start, end in source_fragments
        ):
            rejected_rows.append(
                _rejection(
                    group, "invalid_source_fragment", "outside source anchor bounds"
                )
            )
            continue

        target_size = _as_int(_group_value(group, "t_src_size"))
        target_fragments = [
            (int(start), int(end))
            for start, end in group.select("t_start", "t_end").iter_rows()
        ]
        if any(
            start < 0 or end <= start or end > target_size
            for start, end in target_fragments
        ):
            rejected_rows.append(
                _rejection(group, "invalid_target_bounds", f"t_src_size={target_size}")
            )
            continue
        if group["fragment_id"].n_unique() != group.height:
            rejected_rows.append(
                _rejection(group, "duplicated_mapping", "duplicate fragment IDs")
            )
            continue
        if source_fragments and _has_overlapping_intervals(source_fragments):
            rejected_rows.append(
                _rejection(group, "duplicated_mapping", "overlapping source coverage")
            )
            continue
        if _has_overlapping_intervals(target_fragments):
            rejected_rows.append(
                _rejection(group, "ambiguous_mapping", "overlapping target fragments")
            )
            continue

        pre_resize_start = min(start for start, _ in target_fragments)
        pre_resize_end = max(end for _, end in target_fragments)
        pre_resize_length = pre_resize_end - pre_resize_start
        if pre_resize_length < pre_resize_min_length:
            rejected_rows.append(
                _rejection(
                    group,
                    "span_too_short",
                    f"length={pre_resize_length}; min={pre_resize_min_length}",
                )
            )
            continue
        if pre_resize_length > pre_resize_max_length:
            rejected_rows.append(
                _rejection(
                    group,
                    "span_too_long",
                    f"length={pre_resize_length}; max={pre_resize_max_length}",
                )
            )
            continue
        if target_size < target_length:
            rejected_rows.append(
                _rejection(
                    group,
                    "target_chromosome_too_short",
                    f"t_src_size={target_size}; target_length={target_length}",
                )
            )
            continue

        target_midpoint = (pre_resize_start + pre_resize_end) // 2
        centered_start = target_midpoint - target_length // 2
        centered_end = centered_start + target_length
        if centered_start < 0 or centered_end > target_size:
            rejected_rows.append(
                _rejection(
                    group,
                    "target_window_out_of_bounds",
                    f"centered=[{centered_start}, {centered_end}); t_src_size={target_size}",
                )
            )
            continue

        resized_start, resized_end = resize_to_length(
            pre_resize_start, pre_resize_end, target_length, target_size
        )
        accepted_rows.append(
            {
                "query_name": str(_group_value(group, "query_name")),
                "source_chrom": str(_group_value(group, "source_chrom")),
                "source_start": source_start,
                "source_end": source_end,
                "region_label": str(_group_value(group, "region_label")),
                "species": str(_group_value(group, "species")),
                "alignment_name": str(_group_value(group, "alignment_name")),
                "assembly": str(_group_value(group, "assembly")),
                "taxonomy_id": _as_int(_group_value(group, "taxonomy_id")),
                "family": str(_group_value(group, "family")),
                "clade": str(_group_value(group, "clade")),
                "phylogenetic_rank": _as_int(_group_value(group, "phylogenetic_rank")),
                "alignment_source": str(_group_value(group, "alignment_source")),
                "t_chrom": str(_group_value(group, "t_chrom")),
                "t_start": resized_start,
                "t_end": resized_end,
                "t_strand": str(_group_value(group, "t_strand")),
                "t_src_size": target_size,
                "pre_resize_t_start": pre_resize_start,
                "pre_resize_t_end": pre_resize_end,
                "fragment_count": group.height,
                "aligned_bases": int(group["aligned_bases"].sum()),
            }
        )

    accepted = (
        pl.DataFrame(accepted_rows, schema=ACCEPTED_SCHEMA)
        if accepted_rows
        else pl.DataFrame(schema=ACCEPTED_SCHEMA)
    )
    rejected = (
        pl.DataFrame(rejected_rows, schema=REJECTION_SCHEMA)
        if rejected_rows
        else pl.DataFrame(schema=REJECTION_SCHEMA)
    )
    assert accepted.select("query_name", "species").is_unique().all()
    assert (accepted["t_end"] - accepted["t_start"] == target_length).all()
    assert (accepted["t_start"] >= 0).all()
    assert (accepted["t_end"] <= accepted["t_src_size"]).all()
    accepted_midpoints = (
        accepted["pre_resize_t_start"] + accepted["pre_resize_t_end"]
    ) // 2
    assert (accepted_midpoints - accepted["t_start"] == target_length // 2).all()
    return ProjectionContractResult(
        accepted=accepted.sort("query_name", "species"),
        rejected=rejected.sort("query_name", "species"),
    )


def _apply_single_fragment_contract(
    fragments: pl.DataFrame,
    *,
    target_length: int,
    pre_resize_min_length: int,
    pre_resize_max_length: int,
) -> ProjectionContractResult:
    """Vectorized contract for groups represented by exactly one fragment."""
    if fragments.is_empty():
        return ProjectionContractResult(
            accepted=pl.DataFrame(schema=ACCEPTED_SCHEMA),
            rejected=pl.DataFrame(schema=REJECTION_SCHEMA),
        )
    assert fragments.select("query_name", "species").is_unique().all()

    source_fragment_start = pl.col("source_fragment_start")
    source_fragment_end = pl.col("source_fragment_end")
    has_source_fragment = (
        source_fragment_start.is_not_null() & source_fragment_end.is_not_null()
    )
    invalid_source = has_source_fragment & (
        (source_fragment_start < pl.col("source_start"))
        | (source_fragment_end > pl.col("source_end"))
        | (source_fragment_end <= source_fragment_start)
    )
    invalid_target = (
        (pl.col("t_start") < 0)
        | (pl.col("t_end") <= pl.col("t_start"))
        | (pl.col("t_end") > pl.col("t_src_size"))
    )
    span_length = pl.col("t_end") - pl.col("t_start")
    span_too_short = span_length < pre_resize_min_length
    span_too_long = span_length > pre_resize_max_length
    target_too_short = pl.col("t_src_size") < target_length
    target_midpoint = (pl.col("t_start") + pl.col("t_end")) // 2
    centered_start = target_midpoint - target_length // 2
    centered_end = centered_start + target_length
    target_window_out_of_bounds = (centered_start < 0) | (
        centered_end > pl.col("t_src_size")
    )

    rejection_reason = (
        pl.when(invalid_source)
        .then(pl.lit("invalid_source_fragment"))
        .when(invalid_target)
        .then(pl.lit("invalid_target_bounds"))
        .when(span_too_short)
        .then(pl.lit("span_too_short"))
        .when(span_too_long)
        .then(pl.lit("span_too_long"))
        .when(target_too_short)
        .then(pl.lit("target_chromosome_too_short"))
        .when(target_window_out_of_bounds)
        .then(pl.lit("target_window_out_of_bounds"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    rejection_detail = (
        pl.when(invalid_source)
        .then(pl.lit("outside source anchor bounds"))
        .when(invalid_target)
        .then(
            pl.concat_str(pl.lit("t_src_size="), pl.col("t_src_size").cast(pl.String))
        )
        .when(span_too_short)
        .then(
            pl.concat_str(
                pl.lit("length="),
                span_length.cast(pl.String),
                pl.lit(f"; min={pre_resize_min_length}"),
            )
        )
        .when(span_too_long)
        .then(
            pl.concat_str(
                pl.lit("length="),
                span_length.cast(pl.String),
                pl.lit(f"; max={pre_resize_max_length}"),
            )
        )
        .when(target_too_short)
        .then(
            pl.concat_str(
                pl.lit("t_src_size="),
                pl.col("t_src_size").cast(pl.String),
                pl.lit(f"; target_length={target_length}"),
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
    classified = fragments.with_columns(
        rejection_reason.alias("_rejection_reason"),
        rejection_detail.alias("_rejection_detail"),
    )

    accepted = classified.filter(pl.col("_rejection_reason").is_null()).select(
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "species",
        "alignment_name",
        "assembly",
        "taxonomy_id",
        "family",
        "clade",
        "phylogenetic_rank",
        "alignment_source",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
        pl.col("t_start").alias("pre_resize_t_start"),
        pl.col("t_end").alias("pre_resize_t_end"),
        pl.lit(1, dtype=pl.Int64).alias("fragment_count"),
        "aligned_bases",
    )
    accepted = resize_dataframe(accepted, target_length).cast(ACCEPTED_SCHEMA)
    accepted_midpoints = (
        accepted["pre_resize_t_start"] + accepted["pre_resize_t_end"]
    ) // 2
    assert (accepted_midpoints - accepted["t_start"] == target_length // 2).all()
    rejected = (
        classified.filter(pl.col("_rejection_reason").is_not_null())
        .select(
            "query_name",
            "source_chrom",
            "source_start",
            "source_end",
            "region_label",
            "species",
            "assembly",
            "taxonomy_id",
            "family",
            "clade",
            "phylogenetic_rank",
            "alignment_source",
            pl.col("_rejection_reason").alias("rejection_reason"),
            pl.col("_rejection_detail").alias("detail"),
            pl.lit(1, dtype=pl.Int64).alias("fragment_count"),
        )
        .cast(REJECTION_SCHEMA)
    )
    return ProjectionContractResult(
        accepted=accepted.sort("query_name", "species"),
        rejected=rejected.sort("query_name", "species"),
    )


def _apply_projection_contract_reference(
    fragments: pl.DataFrame,
    *,
    target_length: int = 255,
    pre_resize_min_length: int = 1,
    pre_resize_max_length: int = 2,
) -> ProjectionContractResult:
    """Apply the auditable contract with a vectorized dominant fast path.

    A projection group represented by one fragment cannot have fragment-level
    disagreement or overlap, so those groups are classified and resized using
    Polars expressions. Only genuinely fragmented groups enter the detailed
    per-group overlap checks.
    """
    missing = set(FRAGMENT_SCHEMA) - set(fragments.columns)
    assert not missing, f"projection fragments missing columns: {sorted(missing)}"
    assert target_length > 0
    assert 0 < pre_resize_min_length <= pre_resize_max_length
    if fragments.is_empty():
        return ProjectionContractResult(
            accepted=pl.DataFrame(schema=ACCEPTED_SCHEMA),
            rejected=pl.DataFrame(schema=REJECTION_SCHEMA),
        )

    group_sizes = fragments.group_by("query_name", "species").len()
    classified = fragments.join(group_sizes, on=["query_name", "species"])
    singletons = classified.filter(pl.col("len") == 1).drop("len")
    fragmented = classified.filter(pl.col("len") > 1).drop("len")
    single_result = _apply_single_fragment_contract(
        singletons,
        target_length=target_length,
        pre_resize_min_length=pre_resize_min_length,
        pre_resize_max_length=pre_resize_max_length,
    )
    fragmented_result = _apply_fragmented_projection_contract(
        fragmented,
        target_length=target_length,
        pre_resize_min_length=pre_resize_min_length,
        pre_resize_max_length=pre_resize_max_length,
    )
    accepted = pl.concat([single_result.accepted, fragmented_result.accepted]).sort(
        "query_name", "species"
    )
    rejected = pl.concat([single_result.rejected, fragmented_result.rejected]).sort(
        "query_name", "species"
    )
    assert accepted.height + rejected.height == group_sizes.height
    assert accepted.select("query_name", "species").is_unique().all()
    assert rejected.select("query_name", "species").is_unique().all()
    return ProjectionContractResult(accepted=accepted, rejected=rejected)


def apply_projection_contract(
    fragments: pl.DataFrame,
    *,
    target_length: int = 255,
    pre_resize_min_length: int = 1,
    pre_resize_max_length: int = 2,
) -> ProjectionContractResult:
    """Apply the auditable contract with vectorized group summaries."""
    missing = set(FRAGMENT_SCHEMA) - set(fragments.columns)
    assert not missing, f"projection fragments missing columns: {sorted(missing)}"
    assert target_length > 0
    assert 0 < pre_resize_min_length <= pre_resize_max_length
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
    centered_start = target_midpoint - target_length // 2
    centered_end = centered_start + target_length
    target_window_out_of_bounds = (centered_start < 0) | (
        centered_end > pl.col("t_src_size")
    )
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
        .when(pre_resize_length < pre_resize_min_length)
        .then(pl.lit("span_too_short"))
        .when(pre_resize_length > pre_resize_max_length)
        .then(pl.lit("span_too_long"))
        .when(pl.col("t_src_size") < target_length)
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
        .when(pre_resize_length < pre_resize_min_length)
        .then(
            pl.concat_str(
                pl.lit("length="),
                pre_resize_length.cast(pl.String),
                pl.lit(f"; min={pre_resize_min_length}"),
            )
        )
        .when(pre_resize_length > pre_resize_max_length)
        .then(
            pl.concat_str(
                pl.lit("length="),
                pre_resize_length.cast(pl.String),
                pl.lit(f"; max={pre_resize_max_length}"),
            )
        )
        .when(pl.col("t_src_size") < target_length)
        .then(
            pl.concat_str(
                pl.lit("t_src_size="),
                pl.col("t_src_size").cast(pl.String),
                pl.lit(f"; target_length={target_length}"),
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
    accepted = resize_dataframe(accepted, target_length).cast(ACCEPTED_SCHEMA)
    accepted_midpoints = (
        accepted["pre_resize_t_start"] + accepted["pre_resize_t_end"]
    ) // 2
    assert (accepted_midpoints - accepted["t_start"] == target_length // 2).all()
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


_COMPLEMENT = str.maketrans(
    "ACGTRYSWKMBDHVNacgtryswkmbdhvn",
    "TGCAYRSWMKVHDBNtgcayrswmkvhdbn",
)


def reverse_complement_preserving_case(sequence: str) -> str:
    """Reverse-complement IUPAC DNA while preserving each base's case."""
    return sequence.translate(_COMPLEMENT)[::-1]


def extract_oriented_sequences(
    accepted: pl.DataFrame,
    fetch_sequence: SequenceFetcher,
    *,
    target_length: int = 255,
) -> SequenceExtractionResult:
    """Fetch fixed-length target strings and orient them to the human anchor."""
    missing = set(ACCEPTED_SCHEMA) - set(accepted.columns)
    assert not missing, f"accepted projections missing columns: {sorted(missing)}"
    sequence_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    for row in accepted.to_dicts():
        sequence = fetch_sequence(
            str(row["assembly"]),
            str(row["t_chrom"]),
            int(row["t_start"]),
            int(row["t_end"]),
        )
        if sequence is None or len(sequence) != target_length:
            rejection_rows.append(
                {
                    "query_name": row["query_name"],
                    "source_chrom": row["source_chrom"],
                    "source_start": row["source_start"],
                    "source_end": row["source_end"],
                    "region_label": row["region_label"],
                    "species": row["species"],
                    "assembly": row["assembly"],
                    "taxonomy_id": row["taxonomy_id"],
                    "family": row["family"],
                    "clade": row["clade"],
                    "phylogenetic_rank": row["phylogenetic_rank"],
                    "alignment_source": row["alignment_source"],
                    "rejection_reason": "insufficient_target_sequence",
                    "detail": (
                        "missing sequence"
                        if sequence is None
                        else f"length={len(sequence)}; expected={target_length}"
                    ),
                    "fragment_count": row["fragment_count"],
                }
            )
            continue
        if row["t_strand"] == "-":
            sequence = reverse_complement_preserving_case(sequence)
        else:
            assert row["t_strand"] == "+"
        sequence_rows.append({**row, "sequence": sequence})

    sequence_schema = pl.Schema([*ACCEPTED_SCHEMA.items(), ("sequence", pl.String)])
    with_sequences = (
        pl.DataFrame(sequence_rows, schema=sequence_schema)
        if sequence_rows
        else pl.DataFrame(schema=sequence_schema)
    )
    extraction_rejections = (
        pl.DataFrame(rejection_rows, schema=REJECTION_SCHEMA)
        if rejection_rows
        else pl.DataFrame(schema=REJECTION_SCHEMA)
    )
    assert (with_sequences["sequence"].str.len_bytes() == target_length).all()
    return SequenceExtractionResult(with_sequences, extraction_rejections)
