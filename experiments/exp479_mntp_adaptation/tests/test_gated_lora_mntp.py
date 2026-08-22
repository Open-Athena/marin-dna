from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from exp479_mntp.causal_longrun import LONGRUN_CHECKPOINT_STEPS
from exp479_mntp.gated_lora_mntp import (
    CausalPreservingGatedModel,
    GatedLoraConfig,
    _same_paired_scores,
    _trajectory_tables,
    assert_gated_trainables,
    right_context_use_gate,
)
from exp479_mntp.lora_mntp import LORA_EFFECTIVE_BATCH_SIZE, LORA_TARGET_MODULES


class FakeLoraPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Linear(3, 2, bias=False)
        self.lora_B = nn.Linear(2, 3, bias=False)


class FakePeftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 3, bias=False)
        self.base.requires_grad_(False)
        for target in LORA_TARGET_MODULES:
            setattr(self, target, FakeLoraPair())
        self.adapter_disabled = False

    @contextmanager
    def disable_adapter(self):  # type: ignore[no-untyped-def]
        before = self.adapter_disabled
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = before

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_causal: bool,
        **kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        base = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        if self.adapter_disabled:
            return SimpleNamespace(logits=base)
        adapter_sum = sum(
            parameter.sum() for name, parameter in self.named_parameters() if "lora_B" in name
        )
        context_delta = 0.25 if is_causal else 0.5
        return SimpleNamespace(logits=base + context_delta + adapter_sum * 1e-3)


def test_zero_initialized_gate_is_exactly_causal_and_both_paths_receive_gradients() -> None:
    peft = FakePeftModel()
    model = CausalPreservingGatedModel(peft, vocab_size=3)  # type: ignore[arg-type]
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.ones_like(input_ids)

    mixed, causal, branch, coefficients = model.forward_components(
        input_ids=input_ids,
        attention_mask=attention_mask,
        branch_is_causal=False,
    )

    assert torch.equal(mixed, causal)
    assert torch.count_nonzero(coefficients) == 0
    (mixed.sum() + branch.sum()).backward()
    assert model.mixing_gate.weight.grad is not None
    assert torch.count_nonzero(model.mixing_gate.weight.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in peft.named_parameters()
        if "lora_B" in name
    )


def test_gated_trainable_contract_allows_only_lora_and_seven_value_gate() -> None:
    model = CausalPreservingGatedModel(FakePeftModel(), vocab_size=7)  # type: ignore[arg-type]
    counts = assert_gated_trainables(model)
    assert counts["gate"] == 7
    assert counts["total"] == counts["lora"] + 7


def test_gated_trainable_contract_rejects_unfrozen_base() -> None:
    peft = FakePeftModel()
    peft.base.weight.requires_grad_(True)
    model = CausalPreservingGatedModel(peft, vocab_size=7)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="non-LoRA parameters"):
        assert_gated_trainables(model)


def test_registered_gated_configuration_is_conservative() -> None:
    config = GatedLoraConfig()
    assert config.rank == 16
    assert config.alpha == 16
    assert config.dropout == 0.05
    assert config.mask_probability == 0.2
    assert config.auxiliary_branch_loss_weight == 1.0
    assert config.learning_rate == 1e-5
    assert config.warmup_steps == 100
    assert config.cooldown_start_step == 800
    assert config.train_steps == 1_000
    assert config.microbatch_size * config.accumulation_steps == LORA_EFFECTIVE_BATCH_SIZE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"dropout": 1.0},
        {"mask_probability": 0.0},
        {"auxiliary_branch_loss_weight": 0.0},
        {"learning_rate": 0.0},
        {"microbatch_size": 8, "accumulation_steps": 4},
    ],
)
def test_gated_configuration_rejects_contract_changes(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        GatedLoraConfig(**kwargs)


def _comparison(
    *,
    ce: float,
    ce_low: float,
    ce_high: float,
    accuracy: float,
    accuracy_low: float,
    accuracy_high: float,
) -> dict[str, object]:
    return {
        "candidate": "full",
        "baseline": "causalized",
        "nucleotide_ce_delta": ce,
        "nucleotide_ce_delta_ci95_low": ce_low,
        "nucleotide_ce_delta_ci95_high": ce_high,
        "nucleotide_accuracy_delta": accuracy,
        "nucleotide_accuracy_delta_ci95_low": accuracy_low,
        "nucleotide_accuracy_delta_ci95_high": accuracy_high,
    }


def test_right_context_gate_rejects_a_trivially_closed_gate() -> None:
    gate = right_context_use_gate(
        _comparison(ce=0, ce_low=0, ce_high=0, accuracy=0, accuracy_low=0, accuracy_high=0)
    )
    assert gate["point_noninferior"] is True
    assert gate["confidence_noninferior"] is True
    assert gate["confidence_strict_improvement"] is False
    assert gate["passed"] is False


def test_right_context_gate_accepts_strict_ce_gain_without_accuracy_loss() -> None:
    gate = right_context_use_gate(
        _comparison(
            ce=-0.1,
            ce_low=-0.2,
            ce_high=-0.01,
            accuracy=0,
            accuracy_low=0,
            accuracy_high=0,
        )
    )
    assert gate["passed"] is True


def _scores(readout: str, *, ce: float, correct: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "readout": [readout] * 640,
            "sample_id": list(range(640)),
            "target_nucleotide_index": [index % 255 for index in range(640)],
            "nucleotide_ce": [ce] * 640,
            "nucleotide_correct": [correct] * 640,
            "full_vocab_ce": [ce] * 640,
            "full_vocab_correct": [correct] * 640,
        }
    )


def test_paired_score_identity_is_bit_exact() -> None:
    source = _scores("source", ce=1.0, correct=1.0)
    candidate = source.copy()
    candidate["readout"] = "candidate"
    assert _same_paired_scores(source, candidate)
    candidate.loc[0, "nucleotide_ce"] += 1e-12
    assert not _same_paired_scores(source, candidate)


def test_gated_trajectory_applies_the_final_source_gate() -> None:
    source = _scores("source_causal_adapter_disabled_step0", ce=1.0, correct=0.0)
    frames = []
    for step in LONGRUN_CHECKPOINT_STEPS:
        frame = _scores(
            f"gated_full_step{step:04d}",
            ce=0.5 if step == 1_000 else 1.0,
            correct=1.0 if step == 1_000 else 0.0,
        )
        frame["optimizer_step"] = step
        frames.append(frame)
    callback = SimpleNamespace(
        source_scores=source,
        score_frames=frames,
        saved=set(LONGRUN_CHECKPOINT_STEPS),
    )

    scores, summary, comparisons, gate = _trajectory_tables(callback, n_bootstrap=10)

    assert len(scores) == 640 * (len(LONGRUN_CHECKPOINT_STEPS) + 1)
    assert len(summary) == len(LONGRUN_CHECKPOINT_STEPS) + 1
    assert comparisons["optimizer_step"].tolist() == list(LONGRUN_CHECKPOINT_STEPS)
    assert gate["candidate"] == "gated_full_step1000"
    assert gate["passed"] is True
