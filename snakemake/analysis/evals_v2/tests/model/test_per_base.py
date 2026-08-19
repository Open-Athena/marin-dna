from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from marin_dna_evals.model.scoring import (
    _logits_to_logprobs,
    compute_per_base_stats_clm,
)
from marin_dna_evals.per_base import (
    aggregate_by_case,
    compare_ll_gap_cache,
    compute_hf_per_base_stats,
)
from marin_dna_evals.transforms import transform_ll_clm
from torch import nn


class _FixedModel(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        batch, length = input_ids.shape
        return SimpleNamespace(logits=self.logits[:batch, :length])


def test_per_base_kernel_nll_and_four_nucleotide_entropy() -> None:
    logits = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0, 20.0], [2.0, 2.0, 2.0, 2.0, -5.0], [0, 0, 0, 0, 0]]]
    )
    input_ids = torch.tensor([[4, 2, 1]])
    model = _FixedModel(logits)
    out = compute_per_base_stats_clm(
        model, input_ids, nuc_token_ids=torch.tensor([0, 1, 2, 3])
    )
    expected_nll = -_logits_to_logprobs(logits, input_ids)
    assert out.shape == (1, 2, 2)
    torch.testing.assert_close(out[:, :, 0], expected_nll)
    expected_entropy = torch.tensor(
        [
            torch.distributions.Categorical(logits=logits[0, 0, :4]).entropy(),
            torch.log(torch.tensor(4.0)),
        ]
    )
    torch.testing.assert_close(out[0, :, 1], expected_entropy)


def _sequences() -> pd.DataFrame:
    return pd.DataFrame({"id": ["NC_1:10-14", "NC_2:20-24"], "seq": ["ACgt", "aCGt"]})


def test_compute_hf_per_base_realigns_rc_to_forward_coordinates() -> None:
    fwd = np.arange(16, dtype=np.float32).reshape(2, 4, 2) + 1
    rc = fwd + 100
    mod = "marin_dna_evals.per_base"
    with (
        patch(f"{mod}.AutoTokenizer.from_pretrained", return_value=object()),
        patch(f"{mod}.AutoModelForCausalLM.from_pretrained", return_value=object()),
        patch(f"{mod}._get_special_token_counts", return_value=(1, 0)),
        patch(f"{mod}.run_per_base_stats_clm", side_effect=[fwd, rc]),
    ):
        out = compute_hf_per_base_stats("/unused", _sequences(), 4)
    np.testing.assert_array_equal(np.stack(out["fwd"]["nll"]), fwd[:, :, 0])
    np.testing.assert_array_equal(np.stack(out["rc"]["nll"]), rc[:, ::-1, 0])
    assert list(out["rc"]["window_id"]) == list(_sequences()["id"])


def test_compute_hf_per_base_rejects_non_bos_layout() -> None:
    mod = "marin_dna_evals.per_base"
    with (
        patch(f"{mod}.AutoTokenizer.from_pretrained", return_value=object()),
        patch(f"{mod}.AutoModelForCausalLM.from_pretrained", return_value=object()),
        patch(f"{mod}._get_special_token_counts", return_value=(0, 0)),
        pytest.raises(AssertionError, match="one prepended BOS"),
    ):
        compute_hf_per_base_stats("/unused", _sequences(), 4)


def test_aggregate_and_cache_regression_gate() -> None:
    seq = _sequences()
    atoms = pd.DataFrame(
        {
            "window_id": seq["id"],
            "nll": [
                np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ],
        }
    )
    reconstructed = aggregate_by_case(atoms, seq)
    assert reconstructed.loc[0, "n_upper"] == 2
    assert reconstructed.loc[0, "n_lower"] == 2
    assert reconstructed.loc[0, "ll_sum_upper"] == pytest.approx(-0.3)
    assert reconstructed.loc[0, "ll_sum_lower"] == pytest.approx(-0.7)
    report = compare_ll_gap_cache(reconstructed, reconstructed.copy())
    assert report["passed"] is True
    assert report["max_abs_per_window_sum_diff"] == 0.0


def test_cache_regression_gate_rejects_count_mismatch() -> None:
    seq = _sequences()
    atoms = pd.DataFrame(
        {
            "window_id": seq["id"],
            "nll": [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)],
        }
    )
    reconstructed = aggregate_by_case(atoms, seq)
    bad = reconstructed.copy()
    bad.loc[0, "n_upper"] += 1
    with pytest.raises(AssertionError, match="n_upper mismatch"):
        compare_ll_gap_cache(reconstructed, bad)


class _CharTokenizer:
    bos_token_id = 4
    eos_token_id = None

    def encode(self, sequence: str) -> list[int]:
        token = {"A": 0, "C": 1, "G": 2, "T": 3}
        return [self.bos_token_id, *(token[base] for base in sequence)]


def test_transform_ll_clm_rc_preserves_case_alignment() -> None:
    transformed = transform_ll_clm(
        {"seq": "AaCG"},
        _CharTokenizer(),
        strand="-",
    )
    # reverse-complement("AaCG") == "CGtT"; tokenization sees uppercase,
    # while source case stays attached to its genomic base in RC order.
    torch.testing.assert_close(
        transformed["input_ids"],
        torch.tensor([4, 1, 2, 3, 3]),
    )
    torch.testing.assert_close(
        transformed["is_upper"],
        torch.tensor([False, True, True, False, True]),
    )
