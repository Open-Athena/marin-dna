from __future__ import annotations

import polars as pl
import pytest

from marin_dna.pipelines.vertebrate_projection_dataset.contract import (
    apply_projection_contract,
    extract_oriented_sequences,
    reverse_complement_preserving_case,
)
from marin_dna.pipelines.vertebrate_projection_dataset.maf import FRAGMENT_SCHEMA


def _fragment(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "query_name": "a1",
        "source_chrom": "chr1",
        "source_start": 100,
        "source_end": 104,
        "region_label": "cds",
        "source_fragment_start": 100,
        "source_fragment_end": 104,
        "species": "Gallus gallus",
        "alignment_name": "galGal4",
        "assembly": "Gallus_gallus-4.0",
        "taxonomy_id": 9031,
        "family": "Phasianidae",
        "clade": "birds",
        "phylogenetic_rank": 2,
        "alignment_source": "ucsc_multiz100way",
        "t_chrom": "chr2",
        "t_start": 200,
        "t_end": 204,
        "t_strand": "+",
        "t_src_size": 1000,
        "mapping_id": "m1",
        "fragment_id": "f1",
        "aligned_bases": 4,
    }
    row.update(updates)
    return row


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=FRAGMENT_SCHEMA)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"t_chrom": "chr3"}, "multi_chromosome"),
        ({"t_strand": "-"}, "multi_strand"),
        ({"t_start": -1}, "invalid_target_bounds"),
    ],
)
def test_contract_rejects_conflicting_or_invalid_fragments(
    updates: dict[str, object], reason: str
) -> None:
    second_updates: dict[str, object] = {
        "fragment_id": "f2",
        "source_fragment_start": 102,
        "source_fragment_end": 104,
        "t_start": 202,
    }
    second_updates.update(updates)
    second = _fragment(**second_updates)
    first = _fragment(source_fragment_end=102, t_end=202)
    result = apply_projection_contract(
        _frame(first, second),
        target_length=4,
        pre_resize_min_length=1,
        pre_resize_max_length=10,
    )
    assert result.accepted.is_empty()
    assert result.rejected["rejection_reason"].to_list() == [reason]


def test_contract_rejects_overlapping_source_as_duplicate() -> None:
    result = apply_projection_contract(
        _frame(
            _fragment(source_fragment_end=103, t_end=203),
            _fragment(
                fragment_id="f2",
                source_fragment_start=102,
                t_start=203,
            ),
        ),
        target_length=4,
        pre_resize_min_length=1,
        pre_resize_max_length=10,
    )
    assert result.rejected["rejection_reason"].to_list() == ["duplicated_mapping"]


def test_contract_rejects_pre_resize_length_outside_bounds() -> None:
    result = apply_projection_contract(
        _frame(_fragment(t_end=220)),
        target_length=4,
        pre_resize_min_length=2,
        pre_resize_max_length=10,
    )
    assert result.rejected["rejection_reason"].to_list() == ["span_too_long"]


def test_sequence_orientation_preserves_source_case_and_ignores_conservation() -> None:
    contract = apply_projection_contract(
        _frame(_fragment(t_strand="-")),
        target_length=4,
        pre_resize_min_length=1,
        pre_resize_max_length=10,
    )
    low_score = contract.accepted.with_columns(pl.lit(-100.0).alias("conservation"))
    high_score = contract.accepted.with_columns(pl.lit(100.0).alias("conservation"))

    def fetcher(_assembly: str, _chrom: str, _start: int, _end: int) -> str:
        return "aCgT"

    low = extract_oriented_sequences(low_score, fetcher, target_length=4).accepted
    high = extract_oriented_sequences(high_score, fetcher, target_length=4).accepted
    assert low["sequence"].to_list() == ["AcGt"]
    assert high["sequence"].to_list() == ["AcGt"]
    assert reverse_complement_preserving_case("aCgT") == "AcGt"


def test_sequence_extraction_rejects_short_fetch() -> None:
    contract = apply_projection_contract(
        _frame(_fragment()),
        target_length=4,
        pre_resize_min_length=1,
        pre_resize_max_length=10,
    )
    result = extract_oriented_sequences(
        contract.accepted,
        lambda _assembly, _chrom, _start, _end: "ACG",
        target_length=4,
    )
    assert result.accepted.is_empty()
    assert result.rejected["rejection_reason"].to_list() == [
        "insufficient_target_sequence"
    ]
