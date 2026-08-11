from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.sequence_compatibility import (
    validate_projected_twobit_sizes,
)


def _accepted(path: Path) -> None:
    pl.DataFrame(
        {
            "t_chrom": ["chr1", "chr2", "chr1"],
            "t_src_size": [100, 200, 100],
        }
    ).write_parquet(path)


def test_projected_twobit_sizes_must_match_maf(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.parquet"
    sizes = tmp_path / "sizes.tsv"
    output = tmp_path / "compatibility.json"
    _accepted(accepted)
    sizes.write_text("chr1\t100\nchr2\t200\n")
    validate_projected_twobit_sizes(accepted, sizes, output)
    assert json.loads(output.read_text())["checked_chromosomes"] == 2

    sizes.write_text("chr1\t100\nchr2\t201\n")
    with pytest.raises(AssertionError, match="disagree"):
        validate_projected_twobit_sizes(accepted, sizes, output)


def test_projected_twobit_sizes_reject_missing_chromosome(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.parquet"
    sizes = tmp_path / "sizes.tsv"
    _accepted(accepted)
    sizes.write_text("chr1\t100\n")
    with pytest.raises(AssertionError, match="missing projected chromosomes"):
        validate_projected_twobit_sizes(accepted, sizes, tmp_path / "out.json")
