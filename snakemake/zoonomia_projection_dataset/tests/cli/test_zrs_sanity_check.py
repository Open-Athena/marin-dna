import sys
from pathlib import Path

import polars as pl

from marin_dna_zoonomia_projection.cli.zrs_sanity_check import (
    ZRS_EXPECTATIONS,
    _overlap,
    main,
)


def _write_projection(path: Path, *, include_all: bool) -> None:
    expectations = ZRS_EXPECTATIONS if include_all else ZRS_EXPECTATIONS[:1]
    pl.DataFrame(
        [
            {
                "query_name": query,
                "t_chrom": chrom,
                "t_start": start,
                "t_end": min(start + 255, end),
                "t_strand": "+",
            }
            for query, chrom, start, end in expectations
        ]
    ).write_parquet(path)


def test_overlap_uses_half_open_coordinates() -> None:
    assert _overlap(0, 10, 5, 12) == 5
    assert _overlap(0, 5, 5, 10) == 0


def test_main_writes_pass_report(tmp_path: Path, monkeypatch) -> None:
    parquet = tmp_path / "mouse.parquet"
    report = tmp_path / "report.txt"
    _write_projection(parquet, include_all=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "marin-dna-zrs-check",
            "--mus-parquet",
            str(parquet),
            "--output",
            str(report),
        ],
    )

    assert main() == 0
    assert report.read_text().count("PASS") == len(ZRS_EXPECTATIONS)


def test_main_blocks_incomplete_projection(tmp_path: Path, monkeypatch) -> None:
    parquet = tmp_path / "mouse.parquet"
    report = tmp_path / "report.txt"
    _write_projection(parquet, include_all=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "marin-dna-zrs-check",
            "--mus-parquet",
            str(parquet),
            "--output",
            str(report),
        ],
    )

    assert main() == 1
    assert "no row" in report.read_text()
