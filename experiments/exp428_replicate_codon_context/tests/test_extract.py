from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
import torch

from extract import (
    D_SAE,
    FEATURE_COLUMNS,
    FOCAL_INDEX,
    dense_output_frame,
    selected_feature_values,
    variant_sequences,
    verify_file_hashes,
)


def test_variant_sequences_changes_only_center() -> None:
    reference = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    ref, alt = variant_sequences(reference, "C", "T")
    assert ref == reference and alt[FOCAL_INDEX] == "T"
    assert sum(a != b for a, b in zip(ref, alt, strict=True)) == 1


def test_verify_file_hashes_fails_closed(tmp_path) -> None:
    path = tmp_path / "weights"
    path.write_bytes(b"pinned")
    digest = hashlib.sha256(b"pinned").hexdigest()
    observed = verify_file_hashes(tmp_path, {"weights": digest})
    assert observed["weights"]["sha256"] == digest
    with pytest.raises(AssertionError):
        verify_file_hashes(tmp_path, {"weights": "0" * 64})


def test_selected_feature_values_preserves_requested_order() -> None:
    encoded = torch.zeros((2, D_SAE), dtype=torch.float32)
    encoded[0, 13_637] = 3.5
    encoded[1, 11_064] = 2.0
    selected = selected_feature_values(encoded, [13_637, 11_064])
    np.testing.assert_array_equal(selected, [[3.5, 0.0], [0.0, 2.0]])


def test_dense_output_preserves_zero_and_signed_deltas() -> None:
    values = {
        name: np.asarray([[0.0, 0.0], [float(index + 1), 0.5]], np.float32)
        for index, name in enumerate(FEATURE_COLUMNS)
    }
    output = dense_output_frame(np.asarray([4, 9], np.uint32), values)
    assert output["panel_row"].dtype == pl.UInt32
    assert output["f11064_5m_delta"].to_list() == [0.0, -0.5]
    assert output["f13637_25m_delta"].to_list() == [0.0, -2.5]
    assert all(output[column].dtype == pl.Float32 for column in output.columns[1:])
