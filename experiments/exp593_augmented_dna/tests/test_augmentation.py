from dataclasses import replace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.pipeline import (
    ALPHABET,
    KINDS,
    AugmentationConfig,
    Augmenter,
    Encoder,
    RawWindow,
    collate,
    rc_selected,
    reverse_complement,
)
from exp593_dna.templates import catalog_manifest


@pytest.fixture
def raw():
    return RawWindow("Zmays", "chr1", 100, 20100, "+", ("ACGTacgtNN" * 820)[:8192], 17)


@pytest.fixture
def encoder():
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return Encoder(tokenizer)


def test_full_iupac_rc_preserves_case():
    assert reverse_complement("AaCcGgTtNn") == "nNaAcCgGtT"
    assert reverse_complement(reverse_complement(ALPHABET)) == ALPHABET
    with pytest.raises(ValueError):
        reverse_complement("ACGT-")


@pytest.mark.parametrize("kind", KINDS)
def test_sequence_integrity_and_loss_contracts(kind, raw, encoder):
    probabilities = tuple(float(k == kind) for k in KINDS)
    p = Augmenter(
        encoder, replace(AugmentationConfig(), kind_probabilities=probabilities)
    )
    rc_states = set()
    for occurrence in range(80):
        e = p.build(raw, occurrence)
        a = e.attributes
        rc_states.add(a["rc"])
        source = reverse_complement(raw.seq) if a["rc"] else raw.seq
        source = source[
            a["crop_start_in_oriented_window"] : a["crop_start_in_oriented_window"]
            + a["dna_tokens"]
        ]
        expected = (
            source
            if a["case"] == "native_softmask"
            else source.lower()
            if a["case"] == "unmasked_lower"
            else source.upper()
        )
        roles = {s.role: s.text for s in e.spans if s.role != "text"}
        if kind.endswith("fim"):
            assert roles["left"] + roles["middle"] + roles["right"] == expected
            assert e.scored_tokens == len(roles["middle"]) == a["gap_length"]
            assert len(roles["left"]) >= 128 and len(roles["right"]) >= 128
            assert e.spans[-1].role == "middle"
            if kind == "short_fim":
                assert 1 <= a["gap_length"] <= 3
        else:
            assert roles["dna"] == expected
            assert e.scored_tokens == len(expected) - (kind == "bare_causal")
        if kind == "bare_causal":
            assert not a["metadata_fields"] and not a["mask_description"]
            assert e.text == expected
        target_mask = []
        for s in e.spans:
            n = len(encoder.text_ids(s.text)) if s.role == "text" else len(s.text)
            target_mask.extend([s.score] * n)
        assert np.array_equal(e.loss_weights[:-1] > 0, np.asarray(target_mask[1:]))
        assert e.loss_weights[-1] == 0
        assert np.isclose(e.loss_weights.sum(), 1)
        assert 2048 <= len(e.token_ids) <= e.bucket <= 8192
        assert len(e.token_ids) / e.bucket >= 0.875
    assert rc_states == {True, False}


def test_rc_exact_historical_draw_and_resume(raw, encoder):
    p = Augmenter(encoder)
    for occurrence in [0, 1, 99, 2**32 + 73, 10**12]:
        rng = np.random.default_rng(
            np.random.SeedSequence([472, occurrence & 0xFFFFFFFF, occurrence >> 32])
        )
        assert rc_selected(occurrence) == (rng.random() < 0.5)
        before = p.build(raw, occurrence)
        after = Augmenter(encoder).build(raw, occurrence)
        assert before.attributes == after.attributes
        np.testing.assert_array_equal(before.token_ids, after.token_ids)
        np.testing.assert_array_equal(before.loss_weights, after.loss_weights)


def test_one_window_per_row_and_batch_buckets(raw, encoder):
    p = Augmenter(encoder)
    buckets = set()
    for start in range(0, 64 * 30, 64):
        examples = [p.build(raw, i) for i in range(start, start + 64)]
        assert len({e.bucket for e in examples}) == 1
        buckets.add(examples[0].bucket)
        b = collate(examples)
        assert np.allclose(b["loss_weights"].sum(axis=1), 1)
        assert np.all(b["loss_weights"][b["segment_ids"] == -1] == 0)
        assert set(np.unique(b["segment_ids"])) <= {-1, 0}
        assert np.array_equal((b["segment_ids"] == 0).sum(axis=1), b["lengths"])
    assert buckets == {2048, 4096, 6144, 8192}


def test_configuration_and_source_rejection(raw, encoder):
    with pytest.raises(ValueError):
        AugmentationConfig(case_probabilities=(1,))
    with pytest.raises(ValueError):
        AugmentationConfig(length_buckets=(1024,))
    with pytest.raises(ValueError):
        Augmenter(encoder).build(replace(raw, split="validation"), 0)
    with pytest.raises(ValueError):
        Augmenter(encoder).build(replace(raw, assembly="unknown"), 0)
    with pytest.raises(ValueError):
        replace(raw, seq="ACGT")
    with pytest.raises(ValueError):
        rc_selected(-1)


def test_catalog_is_finite_components_not_whole_examples():
    catalog = catalog_manifest()
    assert catalog["component_count"] >= 500
    for phrase in catalog["components"]["native_softmask"]:
        assert "lower" in phrase.lower() and "upper" not in phrase.lower()
