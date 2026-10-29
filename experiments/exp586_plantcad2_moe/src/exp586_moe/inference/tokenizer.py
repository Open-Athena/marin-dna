"""Restricted-input Hugging Face tokenizer optimizations."""

import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from exp586_moe.inference.contract import validate_sequences


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


def encode_dna(tokenizer: Tokenizer, sequence: str) -> np.ndarray:
    """Encode one complete 8,192-base input through a Hugging Face tokenizer."""
    validate_sequences([sequence])
    ids = np.asarray(
        tokenizer.encode(sequence, add_special_tokens=False).ids, dtype=np.int64
    )
    if ids.shape != (8192,):
        raise ValueError(f"Unexpected token shape: {ids.shape}")
    return ids
