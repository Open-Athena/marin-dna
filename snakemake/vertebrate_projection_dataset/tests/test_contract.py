from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.contract import apply_projection_contract
from marin_dna_vertebrate_projection.maf import FRAGMENT_SCHEMA
from marin_dna_vertebrate_projection.pipeline_io import (
    write_contract_outputs,
)


def _fragment(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "query_name": "a1",
        "source_chrom": "chr1",
        "source_start": 100,
        "source_end": 355,
        "region_label": "cds",
        "source_fragment_start": 227,
        "source_fragment_end": 228,
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
        "t_end": 201,
        "t_strand": "+",
        "t_src_size": 1000,
        "mapping_id": "m1",
        "fragment_id": "f1",
        "aligned_bases": 1,
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
        "source_fragment_start": 228,
        "source_fragment_end": 229,
        "t_start": 201,
        "t_end": 202,
    }
    second_updates.update(updates)
    second = _fragment(**second_updates)
    first = _fragment()
    result = apply_projection_contract(
        _frame(first, second),
    )
    assert result.accepted.is_empty()
    assert result.rejected["rejection_reason"].to_list() == [reason]


def test_contract_rejects_overlapping_source_as_duplicate() -> None:
    result = apply_projection_contract(
        _frame(
            _fragment(source_fragment_end=229),
            _fragment(
                fragment_id="f2",
                source_fragment_start=228,
                source_fragment_end=230,
                t_start=201,
                t_end=202,
            ),
        ),
    )
    assert result.rejected["rejection_reason"].to_list() == ["duplicated_mapping"]


def test_contract_rejects_pre_resize_length_outside_bounds() -> None:
    result = apply_projection_contract(
        _frame(_fragment(t_end=203)),
    )
    assert result.rejected["rejection_reason"].to_list() == ["invalid_projected_span"]


@pytest.mark.parametrize(("start", "end"), [(0, 1), (999, 1_000)])
def test_contract_rejects_target_locus_without_centering_flank(
    start: int, end: int
) -> None:
    fragments = _frame(_fragment(t_start=start, t_end=end, aligned_bases=end - start))

    result = apply_projection_contract(fragments)
    assert result.accepted.is_empty()
    assert result.rejected["rejection_reason"].to_list() == [
        "target_window_out_of_bounds"
    ]


def test_writer_matches_contract(tmp_path: Path) -> None:
    fragments = _frame(
        _fragment(query_name="a3", t_start=300, t_end=301),
        _fragment(query_name="a2"),
        _fragment(query_name="a1"),
        _fragment(
            query_name="a2",
            fragment_id="f2",
            source_fragment_start=228,
            source_fragment_end=229,
            t_chrom="chr3",
            t_start=201,
            t_end=202,
        ),
    )
    expected = apply_projection_contract(fragments)
    fragments_path = tmp_path / "fragments.parquet"
    accepted_path = tmp_path / "accepted.parquet"
    rejected_path = tmp_path / "rejected.parquet"
    fragments.write_parquet(fragments_path)

    write_contract_outputs(
        fragments_path,
        accepted_path,
        rejected_path,
    )

    assert pl.read_parquet(accepted_path).equals(expected.accepted)
    assert pl.read_parquet(rejected_path).equals(expected.rejected)
