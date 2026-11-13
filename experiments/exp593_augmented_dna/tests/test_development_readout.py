import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.development import ConservationSite, canonical_sequence_hash
from exp593_dna.development_readout import (
    binary_ranking_metrics,
    completion_target,
    conservation_examples,
    junction_members,
)
from exp593_dna.pipeline import ALPHABET, Encoder, Span, reverse_complement


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


def test_true_base_has_correct_left_and_right_rc_context_without_target_leakage(
    encoder,
):
    dna = "".join(np.random.default_rng(593).choice(list("ACGT"), 512))
    site = ConservationSite(
        0, "1", 1024, 768, 1280, dna, 1, canonical_sequence_hash(dna), "frozen"
    )
    left, right, fim = conservation_examples(site, encoder)
    for example, target in (
        (left, dna[255].lower()),
        (right, reverse_complement(dna[255]).lower()),
        (fim, dna[255].lower()),
    ):
        prompt, answer = completion_target(example)
        assert answer.tolist() == [encoder.dna_lookup[ord(target)]]
        assert len(prompt) + 1 == len(example.token_ids)
        assert example.scored_tokens == 1
    assert left.text == dna[:256].lower()
    assert right.text == reverse_complement(dna[255:]).lower()
    assert len(completion_target(left)[0]) == 255
    assert len(completion_target(right)[0]) == 256
    assert all(s.role != "text" for e in (left, right) for s in e.spans)
    parts = {s.role: s.text for s in fim.spans if s.role != "text"}
    assert parts["left"] + parts["middle"] + parts["right"] == dna.lower()
    assert "conserved" not in fim.text and "neutral" not in fim.text


def test_ranking_metrics_handle_ties_and_do_not_use_file_order():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    metrics = binary_ranking_metrics(y, scores)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    tied = binary_ranking_metrics(y, np.ones(4))
    assert tied["auroc"] == tied["average_precision"] == 0.5
    for order in (np.array([3, 1, 0, 2]), np.arange(4)[::-1]):
        assert binary_ranking_metrics(y[order], scores[order]) == metrics
        assert binary_ranking_metrics(y[order], np.ones(4)) == tied
    with pytest.raises(ValueError):
        binary_ranking_metrics(np.ones(4), scores)
    with pytest.raises(ValueError):
        binary_ranking_metrics(y, np.array([0, 1, 2, np.nan]))


def test_junction_reencoding_preserves_frozen_blind_context_and_rejects_leakage(
    encoder,
):
    path = (
        Path(__file__).parents[1]
        / "artifacts/development-v1/junction_development_examples.jsonl"
    )
    for line in path.read_text().splitlines():
        record = json.loads(line)
        # The fixture tokenizer has different text lengths from the production
        # tokenizer; the frozen DNA and semantic prompt structure remain exact.
        for member in record["members"]:
            spans = tuple(Span(**s) for s in member["spans"])
            member["attributes"]["input_tokens"] = len(encoder.encode(spans)[0])
        a, b = junction_members(record, encoder)
        assert a.attributes["raw_row"] == b.attributes["raw_row"]
        assert len(completion_target(a)[1]) == len(completion_target(b)[1]) == 2
        record["members"][0]["spans"][0]["text"] = "Recover the donor: [left] "
        with pytest.raises(ValueError, match="presentation changed"):
            junction_members(record, encoder)
