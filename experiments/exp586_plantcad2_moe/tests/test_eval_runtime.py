import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from test_model_contract import tiny_config

from exp586_moe.eval_runtime import NativeScorer, PilotScorer, _prepare_pilot_batch
from exp586_moe.eval_smoke import same_hidden_projection
from experiments.grug.moe_hero_ep.model import Transformer


def test_native_adapter_matches_direct_logits_and_masked_hidden_states():
    # Two examples avoid upstream singleton-batch explicit-mesh restrictions.
    # The full GPU validation supplies the actual dropless kernel and eight devices.
    with jax.set_mesh(compact_grug_mesh()):
        model = Transformer.init(tiny_config(), key=jax.random.key(3))
        adapter = NativeScorer(model, jax.sharding.get_mesh(), pad_to=16)
        adapter.batch_size = 2
        sequences = [[3, 4, 5, 6] * 2, [6, 5, 4, 3] * 2]
        logs, hidden = adapter.evaluate(sequences)
        compact = jnp.asarray(sequences)
        direct = model.logits(compact).astype(jnp.float32)
        expected = jax.nn.log_softmax(direct[:, :-1], axis=-1)
        expected = jnp.take_along_axis(expected, compact[:, 1:, None], axis=-1)[..., 0]
        np.testing.assert_allclose(np.asarray(logs), expected, atol=2e-5)
        expected_hidden, _ = model(compact)
    np.testing.assert_allclose(np.asarray(hidden), expected_hidden, atol=2e-5)


@pytest.mark.parametrize("vocab_size", [8, 5003])
def test_bfloat16_adapter_matches_native_per_token_cross_entropy(vocab_size):
    with jax.set_mesh(compact_grug_mesh()):
        model = jmp.get_policy(
            "params=bfloat16,compute=bfloat16,output=bfloat16"
        ).cast_to_param(
            Transformer.init(
                dataclasses.replace(tiny_config(), vocab_size=vocab_size),
                key=jax.random.key(3),
            )
        )
        adapter = NativeScorer(model, jax.sharding.get_mesh(), pad_to=16)
        adapter.batch_size = 2
        sequences = [[3, 4, 5, 6] * 4, [6, 5, 4, 3] * 4]
        logs, hidden = adapter.evaluate(sequences)
        weights = jnp.ones((2, 16), jnp.float32).at[:, -1].set(0)
        loss = eqx.filter_jit(
            lambda m, x, w: m.next_token_loss(
                x, w, mask=AttentionMask.causal(), reduction="none"
            )
        )(model, jnp.asarray(sequences), weights)
        np.testing.assert_allclose(
            np.asarray(logs), -np.asarray(loss)[:, :-1], atol=1e-6
        )
        direct, projected = same_hidden_projection(
            jnp.asarray(np.asarray(hidden), dtype=model.output_proj.dtype),
            model.output_proj,
            jnp.pad(jnp.asarray(sequences)[:, 1:], ((0, 0), (0, 1))),
            weights,
        )
        np.testing.assert_allclose(np.asarray(direct)[:, :-1], logs, atol=1e-6)
        np.testing.assert_allclose(projected, loss, atol=1e-6)


def test_pilot_scorer_accepts_an_explicit_shorter_context():
    with jax.set_mesh(compact_grug_mesh()):
        model = Transformer.init(tiny_config(), key=jax.random.key(3))
        scorer = PilotScorer(
            model, jax.sharding.get_mesh(), [3, 4, 5, 6], sequence_length=16
        )
        scorer.batch_size = 2
        output = scorer.evaluate(
            [[3, 4, 5, 6] * 2, [6, 5, 4, 3] * 2],
            [[4], [4]],
            [4, 4],
            allele_frequency=False,
        )
        exact = PilotScorer(
            model, jax.sharding.get_mesh(), [3, 4, 5, 6], sequence_length=8
        )
        exact.batch_size = 2
        expected = exact.evaluate(
            [[3, 4, 5, 6] * 2, [6, 5, 4, 3] * 2],
            [[4], [4]],
            [4, 4],
            allele_frequency=False,
        )
    assert output["probabilities"].shape == (2, 10, 4)
    np.testing.assert_allclose(output["probabilities"][:, 1:], 0)
    np.testing.assert_allclose(
        output["probabilities"], expected["probabilities"], atol=2e-5
    )


def test_pilot_query_positions_do_not_change_segment_membership():
    ids, segments, positions, features = _prepare_pilot_batch(
        [[3, 4, 5, 6]],
        [[1, 3]],
        [2],
        batch_size=2,
        sequence_length=8,
    )
    np.testing.assert_array_equal(ids[0], [3, 4, 5, 6, 0, 0, 0, 0])
    np.testing.assert_array_equal(segments[0], [0, 0, 0, 0, -1, -1, -1, -1])
    np.testing.assert_array_equal(positions[0], [1, 3, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(features, [2, 2])
