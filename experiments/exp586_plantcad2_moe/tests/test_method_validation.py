import numpy as np
import pytest

from exp586_moe.method_validation import (
    MethodValidationConfig,
    align_right,
    canonical_window,
    combine,
    metrics,
    parity_probabilities,
    parity_reference_config,
    pooled_inference_config,
    reverse_complement,
    validate_target_sequence,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def encode(self, sequence, *, add_special_tokens):
        assert not add_special_tokens
        return _Encoding(["ACGT".index(base) for base in sequence])


def test_pooled_parity_uses_production_training_capacities():
    config = pooled_inference_config("pretrained")
    assert config.capacity_factor == 32
    assert config.pooled_transport_capacity_factor == 8
    assert config.expert_chunks == 1


def test_language_base_parity_uses_dropless_scatter_reference():
    config, backend, expert_axis_size = parity_reference_config(
        "pretrained", "language-base"
    )
    assert config.moe_implementation == "scatter"
    assert config.report_capacity_overflow
    assert backend == "scatter"
    assert expert_axis_size == 1


def test_dna_trained_parity_retains_completed_run_ep_backend():
    config, backend, expert_axis_size = parity_reference_config(
        "pretrained", "dna-trained"
    )
    assert config.moe_implementation == "fixed_pooled_wave_all_to_all"
    assert config.capacity_factor == 32
    assert config.pooled_transport_capacity_factor == 8
    assert backend == "pooled-ep"
    assert expert_axis_size == 8


def test_method_validation_uses_its_durable_output_for_wandb_artifacts():
    config = MethodValidationConfig(
        condition="pretrained",
        run_id="exp586-method-validation-smoke-test",
        cluster="cw-rno2a",
        checkpoint="s3://example/checkpoint",
        checkpoint_metadata_digest="a" * 64,
        output_prefix=(
            "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/"
            "evaluations/final-v1/validation/test"
        ),
    )
    assert config.artifact_root == config.output_prefix + "/wandb"


def test_intrinsic_sequence_transforms_preserve_target_and_orientation():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("acgtn") == "NACGT"
    assert canonical_window("acgt" * 2048) == "ACGT" * 2048
    assert canonical_window("N" + "A" * 8191).startswith("NA")
    sequence, position = validate_target_sequence("ACG", 2)
    assert sequence == "ACG" and position == 2 and sequence[position] == "G"
    with pytest.raises(ValueError, match="alphabet"):
        reverse_complement("ACGX")


def test_two_sided_combination_maps_complemented_alleles():
    left = np.asarray([[0.4, 0.3, 0.2, 0.1]])
    right = np.asarray([[0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_array_equal(align_right(right), left)
    np.testing.assert_allclose(combine(left, right, 0.5), left, atol=1e-15)
    result = metrics(left, np.asarray([0]), np.asarray([True]))
    assert result["accuracy"] == 1 and result["nll"] == pytest.approx(-np.log(0.4))


def test_parity_probability_inventory_is_split_into_device_batches(monkeypatch):
    calls = []

    class Scorer:
        batch_size = 2

        def __init__(self, model, mesh, base_ids):
            assert base_ids == [0, 1, 2, 3]

        def evaluate(self, ids, positions, feature_positions, *, allele_frequency):
            calls.append(len(ids))
            assert positions == [[1024, 4096, 7168]] * len(ids)
            assert feature_positions == [4096] * len(ids)
            assert not allele_frequency
            return {"probabilities": np.ones((len(ids), 10, 4))}

    monkeypatch.setattr(
        "exp586_moe.method_validation.PilotScorer",
        Scorer,
    )
    windows = ["A" * 8192] * 5
    result = parity_probabilities(object(), object(), _Tokenizer(), windows)
    assert calls == [2, 2, 1]
    assert result.shape == (5, 3, 4)
