from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from marin_dna.pipelines.evals.inference_benchmark import (
    aggregate_harness_llr,
    benchmark_prepared_llr,
    prepare_harness_llr,
)


class _BosDnaTokenizer:
    bos_token_id = 0
    eos_token_id = None
    _ids = {"A": 3, "C": 4, "G": 5, "T": 6}

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *(self._ids[base] for base in text)]


def _harness_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = [
        (100, "A", "G", True),
        (200, "C", "T", False),
    ]
    for pos, ref, alt, target in variants:
        for strand in ("+", "-"):
            strand_ref = ref if strand == "+" else {"A": "T", "C": "G"}[ref]
            strand_alt = alt if strand == "+" else {"G": "C", "T": "A"}[alt]
            rows.append(
                {
                    "chrom": "1",
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "target": target,
                    "subset": "missense_variant",
                    "match_group": 7,
                    "context": "A" * 127,
                    "ref_completion": strand_ref + "C" * 127,
                    "alt_completion": strand_alt + "C" * 127,
                    "strand": strand,
                }
            )
    return pd.DataFrame(rows)


class _TinyCachedCausalLm(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool = False,
        **kwargs: object,
    ) -> SimpleNamespace:
        del past_key_values, kwargs
        logits = input_ids.float().unsqueeze(-1).repeat(1, 1, self.vocab_size)
        logits = logits + torch.arange(
            self.vocab_size, dtype=torch.float, device=input_ids.device
        )
        logits = logits + self.anchor
        out = SimpleNamespace(logits=logits)
        if use_cache:
            batch, length = input_ids.shape
            key = torch.zeros(batch, 1, length, 1, device=input_ids.device)
            value = torch.zeros(batch, 1, length, 1, device=input_ids.device)
            out.past_key_values = ((key, value),)
        return out


def test_prepare_harness_llr_validates_and_tokenizes_once() -> None:
    frame = _harness_frame()
    prepared = prepare_harness_llr(frame, _BosDnaTokenizer())
    assert prepared.input_ids.shape == (4, 256)
    assert prepared.alt_token_id.shape == (4,)
    assert prepared.var_pos == 128
    assert prepared.nuc_token_ids.tolist() == [3, 4, 5, 6]
    assert prepared.metadata["strand"].tolist() == ["+", "-", "+", "-"]


def test_prepare_harness_llr_rejects_non_snv_tail_difference() -> None:
    frame = _harness_frame()
    frame.loc[0, "alt_completion"] = frame.loc[0, "alt_completion"][:-1] + "G"
    with pytest.raises(AssertionError, match="tails differ"):
        prepare_harness_llr(frame, _BosDnaTokenizer())


def test_prepare_harness_llr_rejects_incomplete_strand_pair() -> None:
    frame = _harness_frame().iloc[:-1].reset_index(drop=True)
    with pytest.raises(AssertionError, match="two strand rows"):
        prepare_harness_llr(frame, _BosDnaTokenizer())


def test_aggregate_harness_llr_fwd_rc_minus_average() -> None:
    prepared = prepare_harness_llr(_harness_frame(), _BosDnaTokenizer())
    output = aggregate_harness_llr(
        prepared,
        np.arange(4),
        np.array([1.0, 3.0, 2.0, 4.0]),
    )
    assert len(output) == 2
    np.testing.assert_array_equal(output["llr_fwd"], [1.0, 2.0])
    np.testing.assert_array_equal(output["llr_rc"], [3.0, 4.0])
    np.testing.assert_array_equal(output["minus_llr_avg"], [-2.0, -3.0])


@pytest.mark.parametrize(
    "execution_layout",
    ["prefix-cache", "sequential-branches", "branch-packed", "full-pair"],
)
def test_benchmark_prepared_llr_cpu_padding_and_repetitions(
    execution_layout: str,
) -> None:
    prepared = prepare_harness_llr(_harness_frame(), _BosDnaTokenizer())
    result = benchmark_prepared_llr(
        _TinyCachedCausalLm(),
        prepared,
        batch_size=3,
        device="cpu",
        repetitions=2,
        num_workers=0,
        use_bf16_autocast=False,
        execution_layout=execution_layout,
    )
    assert result.row_indices.tolist() == [0, 1, 2, 3]
    assert result.llr.shape == (4,)
    assert np.isfinite(result.llr).all()
    assert len(result.repeat_seconds) == 2
    assert all(seconds > 0 for seconds in result.repeat_seconds)
    assert result.median_seconds > 0
    assert result.peak_vram_allocated_bytes == 0
    assert result.peak_vram_reserved_bytes == 0
