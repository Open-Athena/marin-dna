"""Verify DNA-only source normalization against the untouched FIM text tokenizer."""

from __future__ import annotations

import hashlib
import json

from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest

from exp593_dna.pipeline import ALPHABET, Encoder

SOURCE_TOKENIZER_SHA256 = (
    "d1cf7a63f3e0e0778a7ef137020e6f39d04be1e6da14fe22d6283eb36cddae5e"
)


def verify_source_tokenizer(encoder: Encoder) -> dict[str, str]:
    """Require identical vocabulary IDs and exact original lowercase DNA tokens.

    The old run hashed a Lowercase/Split-character tokenizer wrapper.
    FIM retains unmodified BPE for task text and forces DNA characters
    directly, so their full tokenizer JSON hashes intentionally differ.
    Verify this relationship rather than disabling the native source check.
    """
    source = pretrained_dna_tokenizer()
    digest = tokenizer_digest(source)
    if digest != SOURCE_TOKENIZER_SHA256:
        raise ValueError("The pinned original 1T DNA tokenizer has changed")
    vocabulary = source.get_vocab()
    if vocabulary != encoder.tokenizer.get_vocab():
        raise ValueError("The source and FIM vocabulary IDs differ")
    for base in ALPHABET:
        if source.encode(base, add_special_tokens=False).ids != [
            int(encoder.dna_lookup[ord(base.lower())])
        ]:
            raise ValueError("Original DNA normalization/character IDs differ")
    vocab_digest = hashlib.sha256(
        json.dumps(vocabulary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "source_tokenizer_sha256": digest,
        "text_tokenizer_sha256": encoder.digest,
        "vocabulary_sha256": vocab_digest,
        "dna_contract": "identical_lowercase_character_ids",
    }
