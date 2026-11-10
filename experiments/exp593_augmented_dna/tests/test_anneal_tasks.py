import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.anneal_tasks import (
    af_alleles,
    combine_strands,
    paired_suffix_llr,
    sv_queries,
    sv_scores,
    task_example,
    true_probabilities,
)
from exp593_dna.pipeline import ALPHABET, Encoder, reverse_complement


@pytest.fixture
def encoder():
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return Encoder(tokenizer)


@pytest.mark.parametrize("fim", [False, True])
@pytest.mark.parametrize("positions", [[4095], [4095, 4096], [4094, 4095, 4096]])
def test_target_shift_and_no_answer_in_fim_flanks(encoder, fim, positions):
    sequence = "c" * 4094 + "ATG" + "c" * 4095
    example, query = task_example(sequence, positions, encoder, fim=fim)
    assert len(example.token_ids) == 8192
    expected = "".join(sequence[p].lower() for p in positions)
    assert example.token_ids[query].tolist() == [
        encoder.dna_lookup[ord(c)] for c in expected
    ]
    if fim:
        parts = {s.role: s.text for s in example.spans if s.role != "text"}
        assert parts["middle"] == expected
        assert parts["left"].endswith(sequence[: positions[0]].lower()[-128:])
        assert parts["right"].startswith(sequence[positions[-1] + 1 :].lower()[:128])
        assert example.loss_weights[query - 1].sum() == pytest.approx(1)
    else:
        assert query.tolist() == positions


def test_strand_combination_aligns_positions_and_complements_before_geomean():
    forward = np.array([[[0.7, 0.1, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4]]])
    other_forward = np.array([[[0.2, 0.4, 0.3, 0.1], [0.4, 0.3, 0.2, 0.1]]])
    reverse = other_forward[:, ::-1, :][:, :, [3, 2, 1, 0]]
    actual = combine_strands(np.log(forward), np.log(reverse))
    expected = np.sqrt(forward * other_forward)
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, (forward + other_forward) / 2)


def test_sv_reference_coordinate_conversion_and_historical_sign():
    assert sv_queries({"left": 100, "right": 110}) == (
        [94, 95, 96, 97, 98, 110, 111, 112, 113, 114],
        list(range(4091, 4101)),
    )
    ref = np.full((1, 10, 4), 0.25)
    alt = ref.copy()
    alt[:, :, 0] = 0.5
    answer = "acgtacgtnn"
    expected = -sum(np.log(2) for c in answer if c == "a") / 10
    assert sv_scores(ref, alt, [answer])[0] == pytest.approx(expected)
    assert true_probabilities(ref, [answer])[0, -2:].tolist() == [0, 0]


def test_af_center_is_not_the_standard_motif_center(encoder):
    row = {"sequence": "c" * 4096 + "A" + "c" * 4095, "ref": "A", "alt": "G"}
    ref, alt, index = af_alleles(row)
    assert index == 1024 and ref[index] == "a" and alt[index] == "g"
    assert sum(a != b for a, b in zip(ref, alt, strict=True)) == 1
    assert reverse_complement(ref)[1023] == "t"
    assert reverse_complement(alt)[1023] == "c"
    example, query = task_example(ref, [index], encoder, fim=True)
    assert len(example.token_ids) == 2048
    assert example.token_ids[query].item() == encoder.dna_lookup[ord("a")]
    with pytest.raises(ValueError, match="reference allele"):
        af_alleles({**row, "ref": "C"})


def test_noncontiguous_fim_is_rejected_but_causal_sv_positions_are_allowed(encoder):
    sequence = "a" * 8192
    with pytest.raises(ValueError, match="contiguous"):
        task_example(sequence, [100, 200], encoder, fim=True)
    _, queries = task_example(sequence, [100, 200], encoder, fim=False)
    assert queries.tolist() == [100, 200]


def test_native_eight_column_head_equals_full_vocabulary_case_mass(
    monkeypatch, encoder
):
    from typing import NamedTuple

    import equinox as eqx
    import jax
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from exp593_dna import anneal_tasks

    class Model(eqx.Module):
        output_proj: jax.Array
        embedding: jax.Array

        def __call__(self, tokens, *, mask):
            return self.embedding.at[tokens].get(
                out_sharding=P("data", None, None)
            ), None

    class Batch(NamedTuple):
        tokens: jax.Array
        attn_mask: None

    class Policy:
        @staticmethod
        def cast_to_compute(model):
            return model

    rng = np.random.default_rng(53)
    mesh = Mesh(
        np.asarray(jax.devices("cpu")).reshape(-1, 1),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    model = Model(
        jax.device_put(
            rng.normal(size=(16, 64)).astype(np.float32),
            NamedSharding(mesh, P("data", "model")),
        ),
        jax.device_put(
            rng.normal(size=(64, 16)).astype(np.float32),
            NamedSharding(mesh, P(None, None)),
        ),
    )
    repeats = len(jax.devices("cpu"))
    batch = Batch(
        jax.device_put(
            np.tile([[1, 5, 7], [3, 9, 11]], (repeats, 1)).astype(np.int32),
            NamedSharding(mesh, P("data", None)),
        ),
        None,
    )
    positions = jax.device_put(
        np.tile([[0, 1], [1, 2]], (repeats, 1)).astype(np.int32),
        NamedSharding(mesh, P("data", None)),
    )
    monkeypatch.setattr(anneal_tasks, "apply_qb_betas", lambda params, pending: params)
    with jax.set_mesh(mesh):
        actual = np.exp(
            anneal_tasks.make_base_step(Policy(), encoder)(
                model, None, batch, positions
            )
        )
    hidden = np.asarray(model.embedding)[np.asarray(batch.tokens)]
    selected = np.take_along_axis(hidden, np.asarray(positions)[:, :, None], axis=1)
    full = np.asarray(jax.nn.softmax(selected @ np.asarray(model.output_proj), axis=-1))
    ids = [encoder.dna_lookup[ord(c)] for c in "ACGTacgt"]
    expected = full[:, :, ids[:4]] + full[:, :, ids[4:]]
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_paired_af_suffix_reduction_keeps_small_allele_effect_order():
    ref = np.full((4, 1024), 10, dtype=np.float32)
    alt = ref.copy()
    alt[:, 0] -= np.asarray([2e-5, 4e-5, 6e-5, 8e-5], dtype=np.float32)
    assert np.all(np.diff(paired_suffix_llr(ref, alt)) > 0)
    assert (ref.sum(axis=1) - alt.sum(axis=1)).tolist() == [0, 0, 0, 0]
