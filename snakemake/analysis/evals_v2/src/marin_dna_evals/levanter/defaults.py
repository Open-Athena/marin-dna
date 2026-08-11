# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DNA-experiment helper functions.

Ported from ``marin-community/marin@dna-dev``: ``experiments/dna/defaults.py``.
Only ``dna_effective_seq_len`` migrates — the V1 dataset constants and the
``dna_tokenize_*`` / ``dna_train`` helpers are tied to V1-era experiments that
stay frozen on dna-dev (per issue #168).
"""

from levanter.tokenizers import load_tokenizer

from marin_dna_evals.levanter.batch_tokenizer import DNABatchTokenizer


def dna_effective_seq_len(base_seq_len: int, tokenizer_name: str) -> int:
    """Compute model context size = base DNA sequence length + special tokens (BOS/EOS).

    Loads the tokenizer to detect which special tokens are defined, so the model
    ``max_seq_len`` stays in sync automatically. Uses ``DNABatchTokenizer`` as the
    single source of truth for special-token detection.
    """
    tok = load_tokenizer(tokenizer_name)
    bt = DNABatchTokenizer(tok)
    return base_seq_len + bt.num_special_tokens
