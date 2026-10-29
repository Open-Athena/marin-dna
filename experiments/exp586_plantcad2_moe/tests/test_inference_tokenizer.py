import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from exp586_moe.inference.tokenizer import (
    compact_dna_tokenizer,
    encode_dna,
    reduced_dna_tokenizer,
)


def _source_tokenizer() -> Tokenizer:
    vocab = {f"token_{index}": index for index in range(100)}
    for base, token_id in zip("acgtn", [64, 66, 70, 83, 77], strict=True):
        del vocab[f"token_{token_id}"]
        vocab[base] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="token_0"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return tokenizer


def test_reduced_tokenizer_preserves_trained_ids_and_case_normalization() -> None:
    source = _source_tokenizer()
    reduced = reduced_dna_tokenizer(source)
    sequence = ("ACGTN" * 1640)[:8192]

    expected = np.asarray(source.encode(sequence).ids)
    actual = encode_dna(reduced, sequence)

    np.testing.assert_array_equal(actual, expected)
    assert reduced.encode("ACGTN").ids == reduced.encode("acgtn").ids
    assert reduced.get_vocab_size() == 84


def test_compact_tokenizer_remaps_acgtn_and_normalizes_case() -> None:
    tokenizer = compact_dna_tokenizer()

    assert tokenizer.encode("ACGTN").ids == [0, 1, 2, 3, 4]
    assert tokenizer.encode("ACGTN").ids == tokenizer.encode("acgtn").ids
    assert tokenizer.get_vocab_size() == 5
