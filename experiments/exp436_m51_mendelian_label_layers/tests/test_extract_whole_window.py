from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from extract_focal import D_SAE, M51_HIDDEN_SIZE, WINDOW_BP
from extract_whole_window import (
    create_matrix,
    matrix_relative,
    mean_sae_code,
    summarize_matrix,
    write_paired_batch,
)


class _FakeSAE:
    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        result = torch.zeros((*raw.shape[:2], D_SAE), dtype=torch.float32)
        result[:, :, 3] = raw[:, :, 0]
        result[:, :, 7] = 2 * raw[:, :, 1]
        return result


def test_mean_sae_code_pools_after_encoding() -> None:
    raw = torch.zeros((1, WINDOW_BP, M51_HIDDEN_SIZE), dtype=torch.float32)
    raw[:, :, 0] = torch.arange(WINDOW_BP, dtype=torch.float32)
    raw[:, :, 1] = 4

    pooled = mean_sae_code(raw, _FakeSAE())  # type: ignore[arg-type]

    assert pooled.shape == (1, D_SAE)
    assert pooled[0, 3].item() == (WINDOW_BP - 1) / 2
    assert pooled[0, 7].item() == 8
    assert torch.count_nonzero(pooled).item() == 2


def test_exact_float32_pair_matrix_round_trip(tmp_path: Path) -> None:
    ref_path = tmp_path / matrix_relative("block19-25m", "forward", "ref")
    alt_path = tmp_path / matrix_relative("block19-25m", "forward", "alt")
    ref = create_matrix(ref_path, rows=2)
    alt = create_matrix(alt_path, rows=2)
    pooled = torch.zeros((4, D_SAE), dtype=torch.float32)
    pooled[0, 1] = 1.25
    pooled[1, 1] = 1.5
    pooled[2, 2] = 2.25
    pooled[3, 2] = 2.75

    write_paired_batch(ref, alt, pooled, offset=0, stop=2)
    ref.flush()
    alt.flush()

    observed_ref = np.load(ref_path, mmap_mode="r")
    observed_alt = np.load(alt_path, mmap_mode="r")
    assert observed_ref.dtype == observed_alt.dtype == np.float32
    np.testing.assert_array_equal(observed_ref[:, 1], [1.25, 0])
    np.testing.assert_array_equal(observed_alt[:, 2], [0, 2.75])
    assert summarize_matrix(ref_path, rows=2)["features_observed"] == 2
