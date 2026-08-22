from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from exp479_mntp.mendelian_reload_audit import (
    REFERENCE_BUCKET,
    REFERENCE_KEY,
    FaiRecord,
    fetch_s3_reference_sequence,
    mendelian_score_parity,
    parse_fai,
    select_mendelian_reload_rows,
)


class _RangeS3:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def get_object(self, *, Bucket: str, Key: str, Range: str) -> dict[str, BytesIO]:
        assert Bucket == REFERENCE_BUCKET
        assert Key == REFERENCE_KEY
        start, end = (int(value) for value in Range.removeprefix("bytes=").split("-"))
        return {"Body": BytesIO(self.body[start : end + 1])}


def test_parse_fai_and_fetch_half_open_sequence_across_lines() -> None:
    fasta = b">1\nACGT\nACGT\nAC\n"
    records = parse_fai("1\t10\t3\t4\t5\n")

    observed = fetch_s3_reference_sequence(
        _RangeS3(fasta),
        records,
        chrom="1",
        start=2,
        end=9,
    )

    assert records["1"] == FaiRecord(length=10, offset=3, line_bases=4, line_width=5)
    assert observed == "GTACGTA"


def test_select_mendelian_rows_keeps_fixed_complete_match_groups() -> None:
    rows = []
    for subset in ("a", "b"):
        for group in range(30):
            for copy in range(4):
                rows.append(
                    {
                        "chrom": "1",
                        "pos": 1_000 + group,
                        "ref": "A",
                        "alt": "C",
                        "label": copy == 0,
                        "subset": subset,
                        "match_group": group,
                        "step_1000": float(group),
                    }
                )

    selected = select_mendelian_reload_rows(pd.DataFrame(rows), groups_per_subset=3)

    assert len(selected) == 24
    assert selected.groupby("subset")["match_group"].nunique().eq(3).all()
    assert selected.groupby(["subset", "match_group"])["label"].nunique().eq(2).all()


def test_mendelian_score_parity_is_paired_and_tolerance_gated() -> None:
    reference = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    passing = mendelian_score_parity(reference, reference + 0.001, tolerance=0.002)
    failing = mendelian_score_parity(reference, reference + 0.003, tolerance=0.002)

    assert passing["passed"] is True
    assert failing["passed"] is False
    with pytest.raises(ValueError, match="paired"):
        mendelian_score_parity(reference, reference[:2])
