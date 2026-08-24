"""No-GPU contract tests for issue #515."""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from glm_experiments.data.evals import transform_llr_clm_odd
from glm_experiments.data.lm_datamodule import build_soft_mask, has_eligible_target
from glm_experiments.exp515.config import (
    ACCELERATOR,
    ALL_IN_CAP_USD,
    BRIDGE_STEPS,
    CANARY_STEPS,
    EFFECTIVE_BATCH_SIZE,
    GPU_COMPUTE_CAP_USD,
    GPU_PRICE_PER_HOUR_USD,
    NUCLEOTIDE_LENGTH,
)
from glm_experiments.exp515.data import (
    SequencePlanDataset,
    sha256_file,
    validate_sequence_plan,
)
from glm_experiments.exp515.diagnostics import selector_composition_counts
from glm_experiments.exp515.evaluation import _prepare_model_for_evaluation
from glm_experiments.exp515.module import learning_rate_factor
from glm_experiments.exp515.runner import (
    _archive_precompletion_failure,
    _completed_bridge_state,
    _retain_for_publication,
    _selector_device_smoke,
)
from glm_experiments.exp515.storage import (
    ISSUE_BUCKET_REGION,
    validate_issue_s3_prefix,
)
from glm_experiments.models.components.lm import CLM
from glm_experiments.models.components.selection import TokenSelector


class IdentityEncoder(nn.Identity):
    is_causal = True


class FakeTokenizer:
    mask_token_id = 9

    def encode(self, text: str) -> list[int]:
        lookup = {"A": 3, "C": 4, "G": 5, "T": 6}
        return [2, *(lookup[value] for value in text)]


class FakeGenome:
    def __call__(self, chrom: str, start: int, end: int) -> str:
        assert chrom == "1"
        assert (start, end) == (2, 7)
        return "AACAA"


def _write_plan(directory: Path, sequences: list[str]) -> None:
    directory.mkdir()
    sequence_path = directory / "sequences.bin"
    species_path = directory / "species.u16"
    sequence_path.write_bytes("".join(sequences).encode("ascii"))
    species_path.write_bytes(b"".join(struct.pack("<H", 0) for _ in sequences))
    manifest = {
        "rows": len(sequences),
        "nucleotide_length": NUCLEOTIDE_LENGTH,
        "species": {"species-a": 0},
        "sequences_sha256": sha256_file(sequence_path),
        "species_sha256": sha256_file(species_path),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_bos_repeat_alignment_and_all_lowercase_filter() -> None:
    sequences = ["ACgtA"]
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 3]])
    observed = build_soft_mask(
        sequences,
        input_ids,
        bos_token_id=2,
        require_leading_bos=True,
    )
    assert torch.equal(
        observed,
        torch.tensor([[False, False, False, True, True, False]]),
    )
    assert has_eligible_target("acgt") is False
    assert has_eligible_target("acGt") is True
    with pytest.raises(ValueError, match="exactly one leading BOS"):
        build_soft_mask(
            sequences,
            input_ids[:, 1:],
            bos_token_id=2,
            require_leading_bos=True,
        )


def test_odd_clm_transform_uses_one_based_boundary_and_one_bos() -> None:
    transformed = transform_llr_clm_odd(
        {"chrom": "1", "pos": 5, "ref": "C", "alt": "G"},
        tokenizer=FakeTokenizer(),
        genome=FakeGenome(),
        window_size=5,
    )
    assert transformed["input_ids"].shape == (2, 6)
    assert transformed["input_ids"][0].tolist() == [2, 3, 3, 4, 3, 3]
    assert transformed["input_ids"][1].tolist() == [2, 3, 3, 5, 3, 3]


def test_sequence_plan_is_fixed_width_and_checksumed(tmp_path: Path) -> None:
    sequences = ["A" * NUCLEOTIDE_LENGTH, "c" + "A" * (NUCLEOTIDE_LENGTH - 1)]
    plan_dir = tmp_path / "plan"
    _write_plan(plan_dir, sequences)
    validate_sequence_plan(plan_dir)
    dataset = SequencePlanDataset(plan_dir, start=1, rows=1)
    assert dataset[0] == {
        "sample_id": 1,
        "sequence": sequences[1],
        "species": "species-a",
    }
    with (plan_dir / "sequences.bin").open("r+b") as handle:
        handle.seek(0)
        handle.write(b"C")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_sequence_plan(plan_dir, force_rehash=True)


def test_selected_loss_is_exact_mean_and_repeats_are_ineligible() -> None:
    torch.manual_seed(3)
    net = CLM(
        nn.Embedding(7, 8),
        IdentityEncoder(),
        nn.Identity(),
        nn.Linear(8, 7),
        selector_enabled=True,
        selector_mode="student_low",
        selector_ratio=0.5,
    )
    input_ids = torch.tensor([[2, 3, 4, 5, 6], [2, 6, 5, 4, 3]])
    soft_masked = torch.tensor(
        [[False, False, True, False, False], [False, True, True, False, False]]
    )
    result = net(
        input_ids,
        input_ids,
        soft_masked,
        0.0,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
    )
    expected = result["loss_per_token"].masked_select(result["selected_mask"]).mean()
    assert torch.equal(result["loss"], expected)
    assert not (result["eligible_mask"] & soft_masked[:, 1:]).any()
    assert int(result["selected_count"]) == 2


def test_uniform_selector_matches_disabled_weighted_loss() -> None:
    torch.manual_seed(7)
    net = CLM(
        nn.Embedding(7, 8),
        IdentityEncoder(),
        nn.Identity(),
        nn.Linear(8, 7),
        selector_enabled=True,
        selector_mode="uniform",
        selector_ratio=1.0,
    )
    input_ids = torch.tensor([[2, 3, 4, 5, 6]])
    soft_masked = torch.tensor([[False, False, True, False, False]])
    enabled = net(input_ids, input_ids, soft_masked, 0.0)
    net.selector_enabled = False
    disabled = net(input_ids, input_ids, soft_masked, 0.0)
    assert torch.allclose(enabled["loss"], disabled["loss"])


def test_empty_batch_has_differentiable_zero_loss() -> None:
    net = CLM(
        nn.Embedding(7, 8),
        IdentityEncoder(),
        nn.Identity(),
        nn.Linear(8, 7),
        selector_enabled=True,
        selector_mode="uniform",
        selector_ratio=1.0,
    )
    input_ids = torch.tensor([[2, 3, 4]])
    result = net(
        input_ids,
        input_ids,
        torch.ones_like(input_ids, dtype=torch.bool),
        0.0,
    )
    assert int(result["eligible_count"]) == 0
    assert int(result["selected_count"]) == 0
    assert result["loss"].item() == 0.0
    result["loss"].backward()


def test_bridge_fork_resets_selector_but_same_arm_resume_restores() -> None:
    losses = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    eligible = torch.ones_like(losses, dtype=torch.bool)
    bridge = TokenSelector(mode="uniform", ratio=1.0, seed=1)
    random_arm = TokenSelector(mode="random", ratio=0.5, seed=17)
    random_arm.load_state_dict(bridge.state_dict())
    expected_first = random_arm(losses, eligible)
    fresh = TokenSelector(mode="random", ratio=0.5, seed=17)
    assert torch.equal(expected_first, fresh(losses, eligible))
    checkpoint = random_arm.state_dict()
    expected_next = random_arm(losses, eligible)
    resumed = TokenSelector(mode="random", ratio=0.5, seed=17)
    resumed.load_state_dict(checkpoint)
    assert torch.equal(resumed(losses, eligible), expected_next)


def test_registered_schedule_and_data_position_contract() -> None:
    assert learning_rate_factor(0, 1000) == pytest.approx(0.01)
    assert learning_rate_factor(99, 1000) == pytest.approx(1.0)
    assert learning_rate_factor(100, 1000) == pytest.approx(1.0)
    assert learning_rate_factor(1100, 1000) == pytest.approx(0.1)
    assert 100 * EFFECTIVE_BATCH_SIZE == 204_800


def test_registered_hardware_and_budget_contract() -> None:
    assert ACCELERATOR == "A100:1"
    assert GPU_PRICE_PER_HOUR_USD == 1.99
    assert GPU_COMPUTE_CAP_USD == 28.0
    assert ALL_IN_CAP_USD == 30.0


def test_composition_diagnostics_cover_registered_dimensions() -> None:
    diagnostic = {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6, 3, 4, 5]]),
        "soft_masked": torch.tensor(
            [[False, False, False, True, False, False, False, False]]
        ),
        "eligible_mask": torch.tensor([[True, True, False, True, True, True, True]]),
        "selected_mask": torch.tensor([[True, False, False, True, False, True, False]]),
        "selection_thresholds": torch.tensor([[0.1, 0.9]]),
    }
    counts = selector_composition_counts(diagnostic)
    dimensions = {dimension for dimension, _ in counts}
    assert dimensions == {
        "target_nucleotide",
        "sequence_position",
        "repeat_boundary_distance",
        "local_gc_fraction",
        "local_7mer_frequency",
    }
    assert counts[("target_nucleotide", "A")] == [2, 1]


def test_issue_storage_prefix_is_scoped_and_versioned() -> None:
    assert validate_issue_s3_prefix(
        "s3://oa-bolinas/issues/515/online-loss-selection/v1"
    ) == ("oa-bolinas", "issues/515/online-loss-selection/v1")
    assert ISSUE_BUCKET_REGION == "us-east-2"
    with pytest.raises(ValueError):
        validate_issue_s3_prefix("s3://oa-bolinas/issues/479/wrong/v1")


def test_publication_retains_evidence_but_excludes_reproducible_caches() -> None:
    assert not _retain_for_publication(Path("source-checkpoint/model.safetensors"))
    assert not _retain_for_publication(Path("evaluation-cache/reference.fa.gz"))
    assert not _retain_for_publication(Path("canary-20/step-20.ckpt"))
    assert not _retain_for_publication(Path("sequence-plan/sequences.bin"))
    assert not _retain_for_publication(Path("sequence-plan/sequences.bin.partial"))
    assert _retain_for_publication(Path("sequence-plan/manifest.json"))
    assert _retain_for_publication(Path("bridge/step-100.ckpt"))
    assert _retain_for_publication(Path("evaluations/bridge.csv"))


def test_selector_device_smoke_contracts_on_cpu() -> None:
    result = _selector_device_smoke(torch.device("cpu"))
    assert result["passed"] is True
    assert result["ranked_masks_passed"] is True
    assert result["random_resume_passed"] is True
    assert result["empty_row_passed"] is True


def test_evaluation_moves_detached_model_to_available_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.parameter = nn.Parameter(torch.zeros(()))
            self.moved_to: torch.device | None = None
            self.training = True

        def parameters(self):
            yield self.parameter

        def to(self, device: torch.device) -> FakeModel:
            self.moved_to = device
            return self

        def eval(self) -> FakeModel:
            self.training = False
            return self

    model = FakeModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = _prepare_model_for_evaluation(model)  # type: ignore[arg-type]
    assert device == torch.device("cuda")
    assert model.moved_to == device
    assert model.training is False


def test_completed_bridge_state_validates_registered_resume(tmp_path: Path) -> None:
    (tmp_path / "canary-20").mkdir()
    (tmp_path / "bridge").mkdir()
    (tmp_path / "smoke-test.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (tmp_path / "canary-20" / "runtime.json").write_text(
        json.dumps({"end_global_step": CANARY_STEPS, "microbatch_size": 128}),
        encoding="utf-8",
    )
    (tmp_path / "bridge" / "runtime.json").write_text(
        json.dumps({"end_global_step": BRIDGE_STEPS, "microbatch_size": 128}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "bridge" / f"step-{BRIDGE_STEPS}.ckpt"
    torch.save({"global_step": BRIDGE_STEPS}, checkpoint)
    observed_checkpoint, microbatch, _, _ = _completed_bridge_state(tmp_path)
    assert observed_checkpoint == checkpoint
    assert microbatch == 128


def test_successful_run_archives_a_precompletion_failure(tmp_path: Path) -> None:
    failure = tmp_path / "failure.json"
    failure.write_text('{"status": "failed"}\n', encoding="utf-8")
    (tmp_path / "final-manifest.json").write_text(
        '{"status": "complete"}\n', encoding="utf-8"
    )
    destination = _archive_precompletion_failure(tmp_path)
    assert destination == tmp_path / "repair-history" / "pre-completion-failure.json"
    assert destination.read_bytes() == b'{"status": "failed"}\n'
    assert not failure.exists()


def test_global_rng_round_trip_reference() -> None:
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    expected = (random.random(), np.random.random(), torch.rand(()))
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    observed = (random.random(), np.random.random(), torch.rand(()))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])
