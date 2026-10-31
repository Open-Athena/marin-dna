import json
import string
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from exp586_moe.data import (
    SCRATCH_VOCAB,
    encode_windows,
    language_tokenizer,
    prepare_window,
    pretrained_dna_tokenizer,
    reverse_complement,
    reverse_complement_selected,
    scratch_tokenizer,
)


def test_pretrained_character_mode_preserves_ids_offsets_and_text_tokenizer() -> None:
    original = language_tokenizer()
    dna = pretrained_dna_tokenizer()
    assert dna.get_vocab() == original.get_vocab()
    assert dna.get_vocab_size() == 128256
    assert original.encode("acgtacgt", add_special_tokens=False).tokens == [
        "ac",
        "gt",
        "ac",
        "gt",
    ]
    restored = Tokenizer.from_str(dna.to_str())
    for sequence in (string.ascii_letters, "ACGT" * 2048, "N" * 8192):
        expected = [original.token_to_id(c) for c in sequence.lower()]
        for tokenizer in (dna, restored):
            encoded = tokenizer.encode(sequence)
            assert encoded.ids == expected
            assert encoded.offsets == [(i, i + 1) for i in range(len(sequence))]
            assert original.decode(encoded.ids) == sequence.lower()
    assert dna.encode("<|end_of_text|>").ids == [128001]
    assert json.loads(original.to_str())["normalizer"] is None
    assert original.encode("a").ids == [128000, 64]


def test_pretrained_data_matches_direct_character_lookup_after_augmentation() -> None:
    original = language_tokenizer()
    dna = pretrained_dna_tokenizer()
    sequence = (string.ascii_letters * 158)[:8192]
    for occurrence in (0, 1, 2, 2**32 + 1):
        batch = encode_windows(dna, [sequence], occurrences=[occurrence])
        expected = prepare_window(sequence, occurrence=occurrence)
        assert batch.token_ids[0].tolist() == [
            original.token_to_id(c) for c in expected
        ]
        assert batch.input_tokens == 8192 and batch.loss_targets == 8191
    with pytest.raises(ValueError, match="one token per base"):
        encode_windows(original, ["a" * 8192])


def test_scratch_matches_historical_tokenizer_for_case_and_ambiguity() -> None:
    old = Tokenizer.from_file(
        str(Path(__file__).parent / "fixtures/exp472_tokenizer.json")
    )
    new = scratch_tokenizer()
    for sequence in ["aCgTnNryswkmbdhv", "ACGT" * 2048, "N" * 8192]:
        assert (
            new.encode(sequence, add_special_tokens=False).ids
            == old.encode(sequence, add_special_tokens=False).ids
        )
    saved = json.loads(new.to_str())
    assert saved["model"]["type"] == "WordLevel"
    restored = Tokenizer.from_str(new.to_str())
    assert restored.encode("AcGt").ids == [3, 4, 5, 6]
    assert restored.encode("[EOS]").ids == [7]


def test_one_window_per_example_without_special_insertion() -> None:
    batch = encode_windows(scratch_tokenizer(), ["a" * 8192, "C" * 8192])
    np.testing.assert_array_equal(batch.token_ids[:, 0], [3, 4])
    np.testing.assert_array_equal(batch.token_ids[:, -1], [3, 4])
    assert batch.input_tokens == 16384
    assert batch.loss_targets == 16382
    assert SCRATCH_VOCAB["[EOS]"] not in batch.token_ids
    np.testing.assert_array_equal(batch.loss_weights[:, -1], [0, 0])


def test_padding_has_no_targets_and_never_truncates() -> None:
    batch = encode_windows(scratch_tokenizer(), ["a" * 8192], padded_length=8200)
    assert batch.input_tokens == 8192
    assert batch.loss_targets == 8191
    np.testing.assert_array_equal(batch.segment_ids[0, 8192:], -1)
    np.testing.assert_array_equal(batch.loss_weights[0, 8191:], 0)
    with pytest.raises(ValueError, match="truncation is forbidden"):
        encode_windows(scratch_tokenizer(), ["a" * 8192], padded_length=8191)


def test_strand_choice_replays_by_absolute_occurrence_across_epochs() -> None:
    indices = [0, 1, 2, 10, 2638656, 2638657, 2**32, 2**32 + 1]
    first = [prepare_window("A" * 8192, occurrence=i) for i in indices]
    resumed = [prepare_window("A" * 8192, occurrence=i) for i in indices[3:]]
    assert first[3:] == resumed
    assert set(first) == {"a" * 8192, "t" * 8192}
    assert prepare_window("A" * 8192, occurrence=None) == "a" * 8192
    for i, output in zip(indices, first, strict=True):
        assert (output[0] == "t") == reverse_complement_selected(i)


def test_reverse_complement_agrees_with_old_token_mapping() -> None:
    tokenizer = scratch_tokenizer()
    text = "ACGTNRYSWKMBDHV"
    ids = np.asarray(tokenizer.encode(text).ids)
    old_complement = np.asarray([0, 1, 2, 6, 5, 4, 3])
    assert (
        tokenizer.encode(reverse_complement(text)).ids
        == old_complement[ids[::-1]].tolist()
    )
    assert reverse_complement(reverse_complement(text)) == text.lower()
