from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.joined_478 import (
    build_joined_windows,
    leave_one_chrom_7mer_nll,
    parse_windows,
)


class _FakeTwoBit:
    def __init__(self, references: dict[str, str]):
        self.references = references
        self.closed = False

    def chroms(self) -> dict[str, int]:
        return {chrom: len(seq) for chrom, seq in self.references.items()}

    def sequence(self, chrom: str, start: int, end: int) -> str:
        return self.references[chrom][start:end]

    def close(self) -> None:
        self.closed = True


def test_parse_windows_enforces_half_open_width() -> None:
    frame = pd.DataFrame({"id": ["NC_1:10-18"], "seq": ["ACGTACGT"]})
    assert parse_windows(frame, window_size=8) == [("NC_1", 10, 18)]
    bad = pd.DataFrame({"id": ["NC_1:10-19"], "seq": ["ACGTACGT"]})
    with pytest.raises(AssertionError, match="0-based half-open"):
        parse_windows(bad, window_size=8)


def test_7mer_control_is_finite_with_one_direction_at_edges() -> None:
    result = leave_one_chrom_7mer_nll(
        ["ACGTACGTACGTAC", "TGCATGCATGCATG"],
        ["NC_1", "NC_2"],
    )
    assert len(result) == 2
    assert all(array.shape == (14,) for array in result)
    assert all(np.isfinite(array).all() for array in result)
    assert all((array >= 0).all() for array in result)


def test_7mer_control_preserves_ambiguous_targets() -> None:
    result = leave_one_chrom_7mer_nll(
        ["ACGTACGNACGTACG", "TGCATGCATGCATGC"],
        ["NC_1", "NC_2"],
    )[0]
    assert np.isnan(result[7])
    assert np.isfinite(result[0])
    assert np.isfinite(result[-1])


def test_repeat_join_preserves_independent_case_masks_and_counts() -> None:
    references = {"NC_1": "NNNNNACgtNNacNNNN"}
    fake = _FakeTwoBit(references)
    module = SimpleNamespace(open=lambda _path, _masked: fake)
    sequences = pd.DataFrame({"id": ["NC_1:5-13"], "seq": ["ACGTnnAC"]})
    with patch.dict(sys.modules, {"py2bit": module}):
        joined, manifest = build_joined_windows(
            sequences,
            region="upstream",
            window_size=8,
            repeat_twobit_path="/unused.2bit",
        )
    assert fake.closed
    np.testing.assert_array_equal(
        joined.loc[0, "is_repeat"],
        [False, False, True, True, False, False, True, True],
    )
    np.testing.assert_array_equal(
        joined.loc[0, "is_conserved"],
        [True, True, True, True, False, False, True, True],
    )
    np.testing.assert_array_equal(
        joined.loc[0, "is_ambiguous"],
        [False, False, False, False, True, True, False, False],
    )
    assert joined.loc[0, "sequence_upper"] == "ACGTNNAC"
    assert joined.loc[0, "window_gc"] == pytest.approx(3 / 6)
    assert manifest["n_repeat_positions"] == 4
    assert manifest["n_conserved_positions"] == 6
    assert manifest["n_ambiguous_positions"] == 2
    assert manifest["secondary_cds_strata"] is False
    assert "codon_position" not in joined


def test_repeat_join_fails_on_assembly_mismatch() -> None:
    fake = _FakeTwoBit({"NC_1": "TTTTTTTT"})
    module = SimpleNamespace(open=lambda _path, _masked: fake)
    sequences = pd.DataFrame({"id": ["NC_1:0-8"], "seq": ["ACGTACGT"]})
    with (
        patch.dict(sys.modules, {"py2bit": module}),
        pytest.raises(AssertionError, match="assembly/coordinate mismatch"),
    ):
        build_joined_windows(
            sequences,
            region="downstream",
            window_size=8,
            repeat_twobit_path="/unused.2bit",
        )
    assert fake.closed


def test_cds_annotation_scope_is_explicit() -> None:
    sequences = pd.DataFrame({"id": ["NC_1:0-8"], "seq": ["ACGTACGT"]})
    with pytest.raises(AssertionError, match="required for CDS"):
        build_joined_windows(
            sequences,
            region="cds",
            window_size=8,
            repeat_twobit_path="/unused.2bit",
        )
    with pytest.raises(AssertionError, match="forbidden for other regions"):
        build_joined_windows(
            sequences,
            region="upstream",
            window_size=8,
            repeat_twobit_path="/unused.2bit",
            cds_gtf_path="/unused.gtf.gz",
        )
