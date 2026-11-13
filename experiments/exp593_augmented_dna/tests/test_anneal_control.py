"""Scientific contract for the original-data 545B annealing control."""

from importlib.resources import files

import jax
import numpy as np
from exp586_moe.data import encode_windows, reverse_complement_selected

from exp593_dna.anneal import REFERENCE_TOKENS, TRIALS
from exp593_dna.anneal_control_batches import OldDistributionSource, prepare
from exp593_dna.anneal_control_train import (
    CONTROL_HORIZON,
    CONTROL_TRIAL,
    TrainingConfig,
)
from exp593_dna.pipeline import RawWindow, load_registry


def test_control_schedule_and_corpus_identity() -> None:
    assert CONTROL_HORIZON == 228_881 * REFERENCE_TOKENS
    assert CONTROL_HORIZON < 120_000_000_000
    assert CONTROL_TRIAL.source_checkpoint == TRIALS["545b"].source_checkpoint
    assert CONTROL_TRIAL.source_metadata_sha256 == TRIALS["545b"].source_metadata_sha256
    assert CONTROL_TRIAL.factor(0) == TRIALS["545b"].factor(0)
    assert CONTROL_TRIAL.factor(CONTROL_TRIAL.ramp_tokens) == TRIALS["545b"].factor(
        TRIALS["545b"].ramp_tokens
    )
    assert abs(CONTROL_TRIAL.factor(CONTROL_HORIZON) - 0.05) < 1e-12
    assert CONTROL_TRIAL.fim_probability(CONTROL_HORIZON) == 0
    assert TrainingConfig(trial="545b-control", cluster="cw-rno2a", nodes=1).identity
    assert (
        files("exp593_dna").joinpath("assets/corpus.json").read_bytes()
        == files("exp586_moe").joinpath("corpus.json").read_bytes()
    )


def test_control_batch_matches_old_loader_exactly() -> None:
    assembly = next(iter(sorted(load_registry())))
    original = ("acgtnACGTN" * 820)[:8192]
    records = [
        RawWindow(assembly, "chr1", i * 8192, (i + 1) * 8192, "+", original, i)
        for i in range(64)
    ]

    class Dataset:
        def get_batch(self, indices):
            return [records[i % len(records)] for i in indices]

    source = OldDistributionSource(Dataset())
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, 1, 1),
        ("replica_dcn", "data", "expert"),
    )
    local = prepare(source, mesh, CONTROL_TRIAL, stage_tokens=0, stage_examples=0)
    occurrences = [CONTROL_TRIAL.source_examples + i for i in range(64)]
    expected = encode_windows(
        source.tokenizer, [r.seq for r in records], occurrences=occurrences
    )
    np.testing.assert_array_equal(local.tokens, expected.token_ids)
    np.testing.assert_array_equal(local.weights, expected.loss_weights)
    np.testing.assert_array_equal(local.segments, expected.segment_ids)
    np.testing.assert_array_equal(local.lengths, expected.lengths)
    assert local.details[:, 6].tolist() == [
        int(reverse_complement_selected(i)) for i in occurrences
    ]
    assert np.all(local.details[:, 4] == 8191)
    assert np.all(local.details[:, 5] == 0)
