from __future__ import annotations

from pathlib import Path

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.contract import (
    apply_projection_contract,
)
from marin_dna.pipelines.vertebrate_projection_dataset.maf import (
    MafSequence,
    iter_maf_blocks,
    project_anchors_from_maf,
)
from tests.pipelines.vertebrate_projection_dataset.helpers import species_manifest


def _anchors() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": ["anchor1"],
            "source_chrom": ["chr1"],
            "source_start": [100],
            "source_end": [110],
            "region_label": ["cds"],
        }
    )


def test_maf_negative_coordinates_convert_to_forward_half_open() -> None:
    row = MafSequence("xenTro7.scaffold_1", 100, 10, "-", 1000, "A" * 10)
    assert row.forward_interval == (890, 900)
    assert row.forward_coordinates() == list(range(899, 889, -1))


def test_maf_gap_and_reverse_strand_projection(tmp_path: Path) -> None:
    maf = tmp_path / "tiny.maf"
    maf.write_text(
        "##maf version=1\n\n"
        "a score=1\n"
        "s hg38.chr1 100 10 + 1000 ACGTACGTAA\n"
        "s galGal4.chr2 200 8 + 1000 ACGT--GTAA\n"
        "s xenTro7.scaf 100 10 - 1000 TTTTTTTTTT\n"
    )
    blocks = list(iter_maf_blocks(maf))
    assert len(blocks) == 1

    fragments = project_anchors_from_maf(maf, _anchors(), species_manifest())
    chicken = fragments.filter(pl.col("alignment_name") == "galGal4")
    assert chicken.height == 2
    assert chicken.select("source_fragment_start", "source_fragment_end").rows() == [
        (100, 104),
        (106, 110),
    ]
    frog = fragments.filter(pl.col("alignment_name") == "xenTro7")
    assert frog.select("t_start", "t_end", "t_strand").row(0) == (890, 900, "-")

    result = apply_projection_contract(
        fragments,
        target_length=8,
        pre_resize_min_length=1,
        pre_resize_max_length=20,
    )
    assert result.accepted.height == 2
    assert result.rejected.is_empty()


def test_anchor_split_across_maf_blocks_is_accepted(tmp_path: Path) -> None:
    maf = tmp_path / "fragmented.maf"
    maf.write_text(
        "##maf version=1\n\n"
        "a score=1\n"
        "s hg38.chr1 100 5 + 1000 AAAAA\n"
        "s galGal4.chr2 200 5 + 1000 CCCCC\n\n"
        "a score=1\n"
        "s hg38.chr1 105 5 + 1000 AAAAA\n"
        "s galGal4.chr2 205 5 + 1000 CCCCC\n"
    )
    fragments = project_anchors_from_maf(maf, _anchors(), species_manifest())
    chicken = fragments.filter(pl.col("alignment_name") == "galGal4")
    assert chicken.height == 2
    result = apply_projection_contract(
        chicken,
        target_length=10,
        pre_resize_min_length=1,
        pre_resize_max_length=20,
    )
    assert result.accepted.height == 1
    assert result.accepted.select("t_start", "t_end").row(0) == (200, 210)
