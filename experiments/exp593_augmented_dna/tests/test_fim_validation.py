from dataclasses import replace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.fim_validation import (
    FIMObservation,
    fim_metrics,
    matched_pair,
    prompt_and_target,
)
from exp593_dna.pipeline import (
    ALPHABET,
    AugmentationConfig,
    Augmenter,
    Encoder,
    Example,
    RawWindow,
    reverse_complement,
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


def test_paired_presentations_share_exact_dna_crop_gap_case_and_orientation(encoder):
    dna = "ACGTacgt" * 1024
    raw = RawWindow("Zmays", "1", 0, 12000, "+", dna, 13, "validation")
    augmenter = Augmenter(encoder, replace(AugmentationConfig(), seed=593))
    kinds = set()
    for occurrence in range(40):
        minimal, formatted = matched_pair(augmenter, raw, occurrence)
        kinds.add(minimal.attributes["kind"])
        assert minimal.attributes["pair_key"] == formatted.attributes["pair_key"]
        assert minimal.attributes["metadata_fields"] == []
        assert "species" in formatted.attributes["metadata_fields"]
        assert formatted.attributes["mask_description"]
        by_role = lambda e: {s.role: s.text for s in e.spans if s.role != "text"}
        assert by_role(minimal) == by_role(formatted)
        native = reverse_complement(dna) if minimal.attributes["rc"] else dna
        start = minimal.attributes["crop_start_in_oriented_window"]
        length = minimal.attributes["dna_tokens"]
        shared = by_role(minimal)
        assert (
            shared["left"] + shared["middle"] + shared["right"]
        ).casefold() == native[start : start + length].casefold()
        for member in (minimal, formatted):
            assert len(member.token_ids) <= member.bucket
            assert np.isclose(member.loss_weights.sum(), 1)
            prompt, target = prompt_and_target(member)
            assert len(prompt) + len(target) == len(member.token_ids)
            assert len(target) == member.attributes["gap_length"]
        assert np.array_equal(
            prompt_and_target(minimal)[1], prompt_and_target(formatted)[1]
        )
    assert kinds == {"short_fim", "long_fim"}
    with pytest.raises(ValueError, match="held-out"):
        matched_pair(augmenter, replace(raw, split="train"), 0)


def test_metrics_use_macro_nll_and_only_autoregressive_outputs_for_exact_match(encoder):
    ids = encoder.dna_lookup
    first = Example(
        np.array([ids[ord(c)] for c in "ACG"]),
        np.array([0.5, 0.5, 0]),
        (),
        2048,
        {
            "kind": "short_fim",
            "gap_length": 2,
            "presentation": "minimal",
            "annotation_guided": False,
        },
    )
    second = Example(
        np.array([ids[ord(c)] for c in "AC"]),
        np.array([1.0, 0]),
        (),
        2048,
        {
            "kind": "short_fim",
            "gap_length": 1,
            "presentation": "formatted",
            "annotation_guided": True,
        },
    )
    observations = [
        FIMObservation(first, np.array([4.0, 4.0]), first.token_ids[1:]),
        FIMObservation(
            second, np.array([2.0]), second.token_ids[1:], np.array([ids[ord("c")]])
        ),
    ]
    result = fim_metrics(observations, encoder)
    assert result["eval/fim_loss"] == 3.0
    assert result["eval/fim_base_accuracy"] == 1
    assert result["eval/fim_examples"] == 2
    assert result["eval/fim_targets"] == 3
    assert result["eval/fim_generated_examples"] == 1
    assert result["eval/fim_exact_match"] == 0
    assert result["eval/fim_dna_exact_match"] == 1
    assert "eval/fim_minimal_exact_match" not in result
    assert result["eval/fim_gap_2_loss"] == 4
    observations[1] = replace(observations[1], generated_tokens=np.array([0]))
    assert fim_metrics(observations, encoder)["eval/fim_dna_exact_match"] == 0
    observations[1] = replace(observations[1], generated_tokens=np.array([1, 2]))
    with pytest.raises(ValueError, match="exactly"):
        fim_metrics(observations, encoder)
