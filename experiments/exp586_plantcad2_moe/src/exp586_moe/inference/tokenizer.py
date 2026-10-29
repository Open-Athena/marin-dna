"""Restricted-input Hugging Face tokenizer optimizations."""

import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from exp586_moe.inference.contract import validate_sequences


def _character_bpe(vocab: dict[str, int]) -> Tokenizer:
    """Tokenize DNA as characters in the BPE model without a split pass."""
    used = set(vocab.values())
    unknown_id = next(index for index in range(len(used) + 1) if index not in used)
    tokenizer = Tokenizer(
        models.BPE(
            vocab={**vocab, "[UNK]": unknown_id},
            merges=[],
            unk_token="[UNK]",
        )
    )
    tokenizer.normalizer = normalizers.Lowercase()
    return tokenizer


def compact_dna_tokenizer() -> Tokenizer:
    """Build a five-token ACGTN tokenizer for remapped embedding tables."""
    tokenizer = Tokenizer(
        models.WordLevel({base: index for index, base in enumerate("acgtn")})
    )
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return tokenizer


def compact_dna_bpe_tokenizer() -> Tokenizer:
    """Build the compact tokenizer without materializing isolated pre-tokens."""
    return _character_bpe({base: index for index, base in enumerate("acgtn")})


def reduced_dna_tokenizer(source: Tokenizer) -> Tokenizer:
    """Build a small HF tokenizer that preserves the trained ACGTN token IDs."""
    trained_ids = {base: source.token_to_id(base.lower()) for base in "ACGTN"}
    if any(token_id is None for token_id in trained_ids.values()):
        raise ValueError("Source tokenizer lacks a trained ACGTN token")
    highest_id = max(int(token_id) for token_id in trained_ids.values())
    vocab = {f"[unused_{index}]": index for index in range(highest_id + 1)}
    for base, token_id_or_none in trained_ids.items():
        token_id = int(token_id_or_none)  # checked above
        del vocab[f"[unused_{token_id}]"]
        vocab[base.lower()] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[unused_0]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return tokenizer


def reduced_dna_bpe_tokenizer(source: Tokenizer) -> Tokenizer:
    """Preserve trained DNA IDs while letting the BPE model split characters."""
    trained_ids = {base.lower(): source.token_to_id(base.lower()) for base in "ACGTN"}
    if any(token_id is None for token_id in trained_ids.values()):
        raise ValueError("Source tokenizer lacks a trained ACGTN token")
    return _character_bpe(
        {base: int(token_id) for base, token_id in trained_ids.items()}
    )


def encode_dna(tokenizer: Tokenizer, sequence: str) -> np.ndarray:
    """Encode one complete 8,192-base input through a Hugging Face tokenizer."""
    validate_sequences([sequence])
    ids = np.asarray(
        tokenizer.encode(sequence, add_special_tokens=False).ids, dtype=np.int64
    )
    if ids.shape != (8192,):
        raise ValueError(f"Unexpected token shape: {ids.shape}")
    return ids


def encode_dna_batch(tokenizer: Tokenizer, sequences: list[str]) -> np.ndarray:
    """Encode a batch with the Hugging Face tokenizer's native Rayon path."""
    validate_sequences(sequences)
    ids = np.asarray(
        [
            encoding.ids
            for encoding in tokenizer.encode_batch(
                sequences,
                add_special_tokens=False,
            )
        ],
        dtype=np.int64,
    )
    if ids.shape != (len(sequences), 8192):
        raise ValueError(f"Unexpected token shape: {ids.shape}")
    return ids


def encode_dna_batch_lists(
    tokenizer: Tokenizer, sequences: list[str]
) -> list[list[int]]:
    """Encode directly to the list representation consumed by vLLM."""
    validate_sequences(sequences)
    ids = [
        encoding.ids
        for encoding in tokenizer.encode_batch(
            sequences,
            add_special_tokens=False,
        )
    ]
    if len(ids) != len(sequences) or any(len(row) != 8192 for row in ids):
        raise ValueError("Unexpected token shape from batched list encoding")
    return ids
