from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from exp479_mntp.lora_reload_audit import (
    assert_reloaded_adapter_contract,
    assert_source_tokenizer_contract,
    paired_score_parity,
)


def _scores(*, ce_delta: float = 0.0, correctness_delta: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [0, 1],
            "target_nucleotide_index": [17, 29],
            "target_base": ["A", "T"],
            "nucleotide_ce": [1.0 + ce_delta, 1.2 + ce_delta],
            "nucleotide_correct": [1.0, 0.0 + correctness_delta],
            "full_vocab_ce": [2.0 + ce_delta, 2.2 + ce_delta],
            "full_vocab_correct": [0.0, 0.0 + correctness_delta],
        }
    )


def test_paired_score_parity_accepts_exact_roundtrip() -> None:
    checks = paired_score_parity(_scores(), _scores(), ce_tolerance=0.0)
    assert checks["passed"] is True
    assert checks["n_targets"] == 2
    assert checks["nucleotide_ce_maximum_absolute_delta"] == 0.0
    assert checks["nucleotide_correctness_mismatches"] == 0


def test_paired_score_parity_rejects_ce_or_correctness_change() -> None:
    ce_checks = paired_score_parity(_scores(), _scores(ce_delta=0.01), ce_tolerance=1e-3)
    correctness_checks = paired_score_parity(
        _scores(),
        _scores(correctness_delta=1.0),
        ce_tolerance=0.0,
    )
    assert ce_checks["passed"] is False
    assert correctness_checks["passed"] is False
    assert correctness_checks["nucleotide_correctness_mismatches"] == 1


def test_paired_score_parity_rejects_changed_target_identity() -> None:
    changed = _scores()
    changed.loc[0, "target_nucleotide_index"] = 18
    with pytest.raises(RuntimeError, match="identical paired targets"):
        paired_score_parity(_scores(), changed, ce_tolerance=0.0)


def test_paired_score_parity_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        paired_score_parity(_scores(), _scores(), ce_tolerance=-1.0)


class _Tokenizer:
    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = None
    all_special_ids = (2, 1, 0)

    def __len__(self) -> int:
        return 7


class _Adapter:
    def __init__(self, *, alpha: int = 16, trainable: bool = False) -> None:
        self.peft_config = {
            "default": SimpleNamespace(
                r=16,
                lora_alpha=alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules={
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                },
                modules_to_save=None,
            )
        }
        self.parameter = SimpleNamespace(requires_grad=trainable)

    def named_parameters(self) -> list[tuple[str, SimpleNamespace]]:
        return [
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                self.parameter,
            )
        ]


def test_reload_contracts_accept_registered_tokenizer_and_frozen_adapter() -> None:
    tokenizer = assert_source_tokenizer_contract(_Tokenizer())
    adapter = assert_reloaded_adapter_contract(_Adapter())
    assert tokenizer["all_special_ids"] == [0, 1, 2]
    assert adapter["alpha"] == 16
    assert adapter["trainable_parameter_names"] == []


def test_reload_contracts_reject_special_id_or_adapter_changes() -> None:
    changed_tokenizer = _Tokenizer()
    changed_tokenizer.unk_token_id = 7
    with pytest.raises(RuntimeError, match="tokenizer contract"):
        assert_source_tokenizer_contract(changed_tokenizer)
    with pytest.raises(RuntimeError, match="adapter contract"):
        assert_reloaded_adapter_contract(_Adapter(alpha=32))
    with pytest.raises(RuntimeError, match="adapter contract"):
        assert_reloaded_adapter_contract(_Adapter(trainable=True))
