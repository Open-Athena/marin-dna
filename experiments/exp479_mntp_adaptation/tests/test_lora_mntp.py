from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from exp479_mntp.causal_longrun import LONGRUN_CHECKPOINT_STEPS
from exp479_mntp.lora_mntp import (
    LORA_ATTENTION_CALIBRATION_CE_DEGRADATION_FRACTIONS,
    LORA_ATTENTION_CALIBRATION_PROBABILITIES,
    LORA_EFFECTIVE_BATCH_SIZE,
    LORA_TARGET_MODULES,
    LoraMntpConfig,
    _evaluate_preserving_mode,
    _trajectory_tables,
    annealed_attention_mask,
    assert_lora_trainables,
    attention_future_edge_probability,
    damage_calibrated_future_edge_probability,
)


class FakeLoraPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Linear(3, 2, bias=False)
        self.lora_B = nn.Linear(2, 3, bias=False)


class FakeLoraModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 3, bias=False)
        self.base.requires_grad_(False)
        for target in LORA_TARGET_MODULES:
            setattr(self, target, FakeLoraPair())


def test_paired_evaluation_moves_to_requested_device_and_restores_training_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeLoraModel()
    model.train()
    bundle = SimpleNamespace(model=model)

    def fake_evaluate(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        assert next(model.parameters()).device.type == "meta"
        model.eval()
        return pd.DataFrame({"sample_id": [0]})

    monkeypatch.setattr("exp479_mntp.lora_mntp.evaluate_readout", fake_evaluate)
    scores = _evaluate_preserving_mode(
        bundle,  # type: ignore[arg-type]
        validation_plan=Path("validation.jsonl"),
        batch_size=1,
        readout="test",
        attention_mode="full",
        evaluation_device="meta",
    )

    assert scores["sample_id"].tolist() == [0]
    assert model.training


def test_selected_lora_configuration_is_single_conservative_pilot() -> None:
    config = LoraMntpConfig()
    assert config.rank == 16
    assert config.alpha == 16
    assert config.dropout == 0.05
    assert config.mask_probability == 0.2
    assert config.attention_anneal_steps == 800
    assert config.learning_rate == 1e-5
    assert config.warmup_steps == 100
    assert config.cooldown_start_step == 800
    assert config.train_steps == 1_000
    assert config.microbatch_size * config.accumulation_steps == LORA_EFFECTIVE_BATCH_SIZE
    assert config.to_dict()["target_modules"] == list(LORA_TARGET_MODULES)


def test_lora_trainable_contract_accepts_only_registered_adapter_matrices() -> None:
    model = FakeLoraModel()
    count, names = assert_lora_trainables(model)
    assert count == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert len(names) == 2 * len(LORA_TARGET_MODULES)
    assert all("lora_A" in name or "lora_B" in name for name in names)


def test_lora_trainable_contract_rejects_unfrozen_base_parameter() -> None:
    model = FakeLoraModel()
    model.base.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="non-LoRA parameters"):
        assert_lora_trainables(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"dropout": 1.0},
        {"mask_probability": 0.0},
        {"attention_anneal_steps": 0},
        {"attention_anneal_steps": 1_001},
        {"learning_rate": 0.0},
        {"microbatch_size": 8, "accumulation_steps": 4},
    ],
)
def test_lora_configuration_rejects_contract_changes(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        LoraMntpConfig(**kwargs)


def test_damage_calibration_inverts_every_measured_knot() -> None:
    for damage_fraction, probability in zip(
        LORA_ATTENTION_CALIBRATION_CE_DEGRADATION_FRACTIONS,
        LORA_ATTENTION_CALIBRATION_PROBABILITIES,
        strict=True,
    ):
        assert damage_calibrated_future_edge_probability(damage_fraction) == pytest.approx(
            probability
        )


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_damage_calibration_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="CE-degradation fraction"):
        damage_calibrated_future_edge_probability(fraction)


def test_attention_annealing_is_monotone_and_leaves_200_full_attention_steps() -> None:
    probabilities = [
        attention_future_edge_probability(step, anneal_steps=800) for step in range(1_000)
    ]
    assert probabilities[0] == 0.0
    assert all(left <= right for left, right in pairwise(probabilities))
    assert sum(probability == 1.0 for probability in probabilities) == 200
    assert probabilities[263] == pytest.approx(0.01, abs=2e-4)
    assert probabilities[393] == pytest.approx(0.02, abs=2e-4)
    assert probabilities[539] == pytest.approx(0.05, abs=2e-4)
    assert probabilities[621] == pytest.approx(0.1, abs=1e-3)
    assert probabilities[800] == 1.0
    assert probabilities[999] == 1.0


def test_annealed_attention_mask_has_exact_causal_and_full_endpoints() -> None:
    tokens = torch.ones((2, 4), dtype=torch.long)
    tokens[1, -1] = 0
    causal = annealed_attention_mask(
        tokens,
        future_edge_probability=0.0,
        seed=11,
        dtype=torch.float32,
    )
    full = annealed_attention_mask(
        tokens,
        future_edge_probability=1.0,
        seed=11,
        dtype=torch.float32,
    )
    expected_causal = torch.ones((4, 4), dtype=torch.bool).tril()
    assert torch.equal(causal[0, 0] == 0, expected_causal)
    assert torch.all(causal[1, 0, :, -1] == torch.finfo(torch.float32).min)
    assert torch.all(full[0] == 0)
    assert torch.all(full[1, 0, :, -1] == torch.finfo(torch.float32).min)


def test_annealed_attention_mask_is_seeded_and_shared_across_batch() -> None:
    tokens = torch.ones((2, 16), dtype=torch.long)
    first = annealed_attention_mask(
        tokens,
        future_edge_probability=0.5,
        seed=21,
        dtype=torch.float32,
    )
    repeated = annealed_attention_mask(
        tokens,
        future_edge_probability=0.5,
        seed=21,
        dtype=torch.float32,
    )
    changed = annealed_attention_mask(
        tokens,
        future_edge_probability=0.5,
        seed=22,
        dtype=torch.float32,
    )
    assert torch.equal(first, repeated)
    assert torch.equal(first[0], first[1])
    assert not torch.equal(first, changed)


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


def _trajectory_callback() -> SimpleNamespace:
    baseline = _scores(
        "source_causal_adapter_disabled_step0",
        ce=1.0,
        correct=0.0,
    )
    frames = []
    for step in LONGRUN_CHECKPOINT_STEPS:
        frame = _scores(
            f"lora_full_step{step:04d}",
            ce=0.5 if step == 1_000 else 1.5,
            correct=1.0 if step == 1_000 else 0.0,
        )
        frame["optimizer_step"] = step
        frames.append(frame)
    return SimpleNamespace(
        source_scores=baseline,
        score_frames=frames,
        saved=set(LONGRUN_CHECKPOINT_STEPS),
    )


def test_lora_trajectory_table_applies_the_final_paired_information_gate() -> None:
    scores, summary, comparisons, gate = _trajectory_tables(
        _trajectory_callback(),
        n_bootstrap=10,
    )
    assert len(scores) == 640 * (len(LONGRUN_CHECKPOINT_STEPS) + 1)
    assert len(summary) == len(LONGRUN_CHECKPOINT_STEPS) + 1
    assert comparisons["optimizer_step"].tolist() == list(LONGRUN_CHECKPOINT_STEPS)
    assert gate["candidate"] == "lora_full_step1000"
    assert gate["passed"] is True


def test_lora_trajectory_rejects_a_changed_target_index() -> None:
    callback = _trajectory_callback()
    callback.score_frames[-1].loc[0, "target_nucleotide_index"] = 254
    if callback.source_scores.loc[0, "target_nucleotide_index"] == 254:
        callback.score_frames[-1].loc[0, "target_nucleotide_index"] = 253
    with pytest.raises(RuntimeError, match="identical sample/target pairs"):
        _trajectory_tables(callback, n_bootstrap=10)
