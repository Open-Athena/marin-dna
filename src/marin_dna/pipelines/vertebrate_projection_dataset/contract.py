"""Shared projection acceptance, resizing, and sequence-extraction contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from marin_dna.pipelines.projection.resize import resize_to_length
from marin_dna.pipelines.vertebrate_projection_dataset.maf import FRAGMENT_SCHEMA


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


def apply_projection_contract(
    fragments: pl.DataFrame,
    *,
    target_length: int = 255,
    pre_resize_min_length: int = 128,
    pre_resize_max_length: int = 512,
) -> ProjectionContractResult:
    """Apply one auditable acceptance policy to backend-native fragments.

    Coordinates entering this function must already be 0-based and half-open.
    Fragmented mappings are accepted only when every fragment agrees on target
    chromosome, strand, metadata, and bounds.  Overlapping source or target
    fragments are treated as duplicated/ambiguous mappings.
    """
    missing = set(FRAGMENT_SCHEMA) - set(fragments.columns)
    assert not missing, f"projection fragments missing columns: {sorted(missing)}"
    assert target_length > 0
    assert 0 < pre_resize_min_length <= target_length <= pre_resize_max_length

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
    return ProjectionContractResult(
        accepted=accepted.sort("query_name", "species"),
        rejected=rejected.sort("query_name", "species"),
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
