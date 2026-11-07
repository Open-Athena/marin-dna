import hashlib
from dataclasses import asdict, replace

import jax
import numpy as np
import pytest
from exp586_moe.data_parallel import replicated_mesh
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.evaluate import Evaluator, binary_batch, causal_example, frozen_fim
from exp593_dna.fim_validation import matched_pair, prompt_and_target
from exp593_dna.pipeline import (
    ALPHABET,
    AugmentationConfig,
    Augmenter,
    Encoder,
    RawWindow,
)


@pytest.fixture
def encoder():
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return Encoder(tokenizer)


def test_frozen_panel_checks_exact_native_source_and_geometry(encoder):
    raw = RawWindow("Zmays", "1", 0, 12000, "-", "ACGTacgt" * 1024, 13, "validation")
    augmenter = Augmenter(encoder, replace(AugmentationConfig(), seed=593))
    cases = set()
    orientations = set()
    for occurrence in range(30):
        pair = matched_pair(augmenter, raw, occurrence)
        record = {
            "source_row": raw.row,
            "source_sequence_sha256": hashlib.sha256(raw.seq.encode()).hexdigest(),
            "members": [
                {
                    "attributes": e.attributes,
                    "bucket": e.bucket,
                    "spans": [asdict(s) for s in e.spans],
                }
                for e in pair
            ],
        }
        reconstructed = frozen_fim(record, raw, encoder)
        for expected, actual in zip(pair, reconstructed, strict=True):
            assert np.array_equal(expected.token_ids, actual.token_ids)
            assert np.array_equal(expected.loss_weights, actual.loss_weights)
        cases.add(pair[0].attributes["case"])
        orientations.add(pair[0].attributes["rc"])
        with pytest.raises(ValueError, match="source identity"):
            frozen_fim(record, replace(raw, seq="TGCAacgt" * 1024), encoder)
        record["members"][0]["attributes"] = dict(pair[0].attributes, gap_start=-1)
        with pytest.raises(ValueError, match="geometry"):
            frozen_fim(record, raw, encoder)
    assert len(cases) == 3 and orientations == {False, True}


def test_validation_weights_are_binary_and_padding_rows_have_no_loss(encoder):
    raw = RawWindow("Zmays", "1", 0, 12000, "+", "ACGTacgt" * 1024, 13, "validation")
    example = causal_example(raw, encoder)
    assert np.array_equal(
        example.token_ids,
        encoder.dna_lookup[np.frombuffer(raw.seq.lower().encode(), dtype=np.uint8)],
    )
    mesh = replicated_mesh()
    with jax.set_mesh(mesh):
        batch = binary_batch([example], mesh)
        weights = np.asarray(batch.loss_weight)
        assert weights[0, :-1].sum() == 8191
        assert weights[0, -1] == 0
        assert not weights[1:].any()
        assert np.array_equal(np.asarray(batch.tokens)[0], np.asarray(batch.tokens)[1])


def test_greedy_decode_never_sees_teacher_forced_gap_prefix(encoder):
    raw = RawWindow("Zmays", "1", 0, 12000, "+", "ACGTacgt" * 1024, 13, "validation")
    augmenter = Augmenter(encoder, AugmentationConfig())
    pair = next(
        pair
        for occurrence in range(100)
        if (pair := matched_pair(augmenter, raw, occurrence))[0].attributes[
            "gap_length"
        ]
        == 3
    )
    assert pair[0].attributes["gap_length"] == 3
    mesh = replicated_mesh()
    engine = object.__new__(Evaluator)
    engine.mesh = mesh
    engine.progress = lambda stage, completed, total: None
    predicted = int(encoder.dna_lookup[ord("N")])
    calls = []

    def greedy(params, pending, batch, positions):
        inputs = np.asarray(batch.tokens)
        ends = np.asarray(positions)
        offset = len(calls)
        for row, example in enumerate(pair):
            prompt, _ = prompt_and_target(example)
            assert ends[row] == len(prompt) + offset - 1
            assert inputs[row, : len(prompt)].tolist() == prompt.tolist()
            assert (
                inputs[row, len(prompt) : len(prompt) + offset].tolist()
                == [predicted] * offset
            )
            assert not inputs[row, len(prompt) + offset :].any()
        calls.append(1)
        return engine._put(np.full(64, predicted, dtype=np.int32))

    engine.greedy_step = greedy
    with jax.set_mesh(mesh):
        outputs = engine.generate(
            type("State", (), {"params": None, "pending_qb_betas": None})(), pair
        )
    assert len(calls) == 3
    assert [o.tolist() for o in outputs] == [[predicted] * 3] * 2


def test_raw_causal_nll_uses_the_native_unreduced_loss_api(monkeypatch):
    import equinox as eqx
    import jax.numpy as jnp
    import jmp
    from exp586_moe.data import WindowBatch
    from exp586_moe.gpu_smoke import device_batch
    from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

    import exp593_dna.evaluate as module

    class TinyModel(eqx.Module):
        projection: jax.Array

        @property
        def output_proj(self):
            return self.projection

        def __call__(self, tokens, *, mask):
            return jax.nn.one_hot(tokens, 4), {}

        def next_token_loss(
            self, tokens, weights, *, mask, reduction, logsumexp_weight
        ):
            hidden = jax.nn.one_hot(tokens, 4)
            labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
            return fused_linear_softmax_cross_entropy_loss(
                hidden, self.projection, labels, weight=weights, reduction=reduction
            )

    monkeypatch.setattr(module, "apply_qb_betas", lambda params, pending: params)
    mesh = replicated_mesh()
    with jax.set_mesh(mesh):
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        model = TinyModel(
            jax.device_put(jnp.eye(4), NamedSharding(mesh, P(None, "model")))
        )
        tokens = np.tile(np.asarray([0, 1, 2, 3], dtype=np.int32), (64, 1))
        weights = np.tile(np.asarray([1, 1, 1, 0], dtype=np.float32), (64, 1))
        batch = device_batch(
            WindowBatch(tokens, np.zeros_like(tokens), weights, np.full(64, 4)), mesh
        )
        causal, fim, greedy = module.make_evaluator(
            jmp.get_policy("params=float32,compute=float32,output=float32")
        )
        actual = np.asarray(causal(model, None, batch))
        assert actual.shape == (64, 4)
        np.testing.assert_allclose(actual[:, :3], np.log(np.e + 3), rtol=1e-6)
        assert not actual[:, -1].any()
        traditional = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=None,
        )
        np.testing.assert_allclose(actual[:, :3].mean(), float(traditional), rtol=1e-6)
        engine = object.__new__(Evaluator)
        engine.mesh = mesh
        positions = engine._put(np.tile(np.arange(3, dtype=np.int32), (64, 1)))
        targets = engine._put(tokens[:, 1:])
        nll, predictions = fim(model, None, batch, positions, targets)
        np.testing.assert_allclose(np.asarray(nll), actual[:, :3], rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(predictions), tokens[:, :-1])
        predictions = greedy(
            model, None, batch, engine._put(np.full(64, 3, dtype=np.int32))
        )
        np.testing.assert_array_equal(np.asarray(predictions), np.full(64, 3))
