import hashlib
from dataclasses import replace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.anneal import TRIALS, terminal_lengths
from exp593_dna.anneal_data import (
    GuidedAugmenter,
    Landmark,
    Presentation,
    RowAnnotations,
)
from exp593_dna.fim_validation import prompt_and_target
from exp593_dna.pipeline import ALPHABET, Encoder, RawWindow, reverse_complement


@pytest.fixture
def encoder():
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return Encoder(tokenizer)


@pytest.fixture
def source():
    raw = RawWindow("Zmays", "chr1", 0, 8192, "+", ("ACGTacgtNn" * 820)[:8192], 17)
    annotation = RowAnnotations(
        raw.row,
        hashlib.sha256(raw.seq.encode()).hexdigest(),
        (
            Landmark(4094, 4097, "translation_start", "training-gff"),
            Landmark(6000, 6002, "splice_donor", "training-gff"),
        ),
    )
    return raw, annotation


@pytest.mark.parametrize("trial", TRIALS.values())
def test_stage_lr_and_fim_endpoints(trial):
    assert trial.factor(0) == trial.factor(trial.ramp_tokens) == trial.inherited_factor
    assert trial.fim_probability(0) == 0
    assert trial.fim_probability(trial.ramp_tokens // 2) == pytest.approx(0.15)
    assert trial.fim_probability(trial.ramp_tokens) == 0.30
    assert trial.fim_probability(trial.horizon) == 0.30
    assert trial.factor(trial.horizon) == pytest.approx(0.05)
    assert trial.factor(
        trial.ramp_tokens + (trial.horizon - trial.ramp_tokens) // 2
    ) == pytest.approx((trial.inherited_factor + 0.05) / 2)
    assert trial.source_tokens == trial.source_update * 524288
    assert trial.source_examples == trial.source_update * 64
    assert trial.horizon <= 216_000_000_000
    assert "curriculum_end" in trial.events(trial.ramp_tokens - 1, trial.ramp_tokens)
    with pytest.raises(ValueError):
        trial.factor(-1)


def test_cooldown_slopes_match_to_rounding():
    slopes = [
        (t.inherited_factor - 0.05) / (t.horizon - t.ramp_tokens)
        for t in TRIALS.values()
    ]
    assert max(slopes) / min(slopes) < 1.00001


@pytest.mark.parametrize(
    "remaining", [131072, 131073, 262145, 524288, 524289, 655359, 655360, 800000]
)
def test_terminal_budget_never_creates_too_small_examples(remaining):
    lengths = terminal_lengths(remaining)
    if lengths is None:
        assert remaining >= 655360
    else:
        assert len(lengths) == 64 and 2048 <= min(lengths) <= max(lengths) <= 8192
        left = remaining - sum(lengths)
        assert left == 0 or left >= 64 * 2048
        if left:
            assert sum(terminal_lengths(left)) == left


@pytest.mark.parametrize("rc", [False, True])
@pytest.mark.parametrize("gap", [1, 2, 3, 16, 32, 64, 128, 256, 512])
def test_guided_geometry_and_objective(encoder, source, rc, gap):
    raw, annotation = source
    augmenter = GuidedAugmenter(encoder)
    modes = ("overlap", "nearby") if gap <= 3 else ("overlap",)
    for mode in modes:
        p = Presentation(71, 4096, 4000, True, gap, mode, rc)
        example = augmenter.build(raw, p, annotation)
        a = example.attributes
        dna = {s.role: s.text for s in example.spans if s.role != "text"}
        original = reverse_complement(raw.seq) if rc else raw.seq
        crop = a["crop_start_in_oriented_window"]
        assert (
            dna["left"] + dna["middle"] + dna["right"]
            == original[crop : crop + a["dna_tokens"]].lower()
        )
        assert len(dna["left"]) >= 128 and len(dna["right"]) >= 128
        assert a["annotation_guided"] and not a["metadata_fields"]
        assert np.isclose(example.loss_weights.sum(), 1)
        assert example.scored_tokens == gap
        assert all(s.text == s.text.lower() for s in example.spans if s.role != "text")
        _, target = prompt_and_target(example)
        assert len(target) == gap
        assert "Zmays" not in example.text and "training-gff" not in example.text
        feature = next(f for f in annotation.landmarks if f.kind == a["landmark_class"])
        start, end = (
            (8192 - feature.end, 8192 - feature.start)
            if rc
            else (feature.start, feature.end)
        )
        lo = crop + a["gap_start"]
        if mode == "overlap":
            assert min(lo + gap, end) > max(lo, start)
        else:
            assert lo + gap <= start or lo >= end
            assert lo >= start - 32 and lo + gap <= end + 32


def test_causal_has_only_lowercase_original_dna_and_rc(encoder, source):
    raw, _ = source
    augmenter = GuidedAugmenter(encoder)
    observed = set()
    for occurrence in range(100):
        p = augmenter.presentation(occurrence, 0)
        example = augmenter.build(raw, p)
        observed.add(p.rc)
        assert len(example.spans) == 1 and example.spans[0].role == "dna"
        assert example.text == example.text.lower()
        assert not example.attributes["annotation_guided"]
    assert observed == {False, True}


def test_fim_requires_matching_annotations_and_train_split(encoder, source):
    raw, annotation = source
    augmenter = GuidedAugmenter(encoder)
    p = Presentation(0, 2048, 2048, True, 2, "overlap", False)
    with pytest.raises(ValueError):
        augmenter.build(raw, p)
    with pytest.raises(ValueError):
        augmenter.build(replace(raw, split="validation"), p, annotation)
    with pytest.raises(ValueError):
        augmenter.build(raw, p, replace(annotation, sequence_sha256="0" * 64))
