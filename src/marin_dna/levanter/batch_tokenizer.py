# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter ``BatchProcessor`` for case-aware DNA tokenization.

Ported from ``marin-community/marin@dna-dev``:
``lib/levanter/src/levanter/data/text/_batch_tokenizer.py``.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from levanter.data import BatchProcessor
from levanter.tokenizers import MarinTokenizer
from levanter.utils.py_utils import logical_cpu_core_count


class DNABatchTokenizer(BatchProcessor[dict, dict]):
    """
    A batch processor that tokenizes DNA sequences with case-based loss weighting.

    Weights are **target-aligned**: ``loss_weight[i]`` reflects the case of the
    *next* token (``input_ids[i+1]``), which is the prediction target at position
    ``i`` in causal LM training.

    Character case determines the weight:
    - Uppercase target (ACGT): weight = uppercase_weight
    - Lowercase target (acgt): weight = lowercase_weight

    If the tokenizer defines BOS/EOS token IDs, they are automatically prepended/appended
    to the token sequences. Use ``num_special_tokens`` to query how many extra tokens
    are added (useful for computing model context size).

    Assumptions:
    - Character-level tokenizer (1:1 character-to-token mapping)
    - All sequences have the same length (no padding/truncation)
    - Model context size matches sequence length + special tokens (see experiment configs).
    """

    def __init__(
        self,
        tokenizer: MarinTokenizer,
        text_field: str = "seq",
        uppercase_weight: float = 1.0,
        lowercase_weight: float = 1.0,
        *,
        override_resources: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self._hf_tokenizer = tokenizer.as_hf_tokenizer()
        self.text_field = text_field
        self.override_resources = override_resources
        self.uppercase_weight = uppercase_weight
        self.lowercase_weight = lowercase_weight
        self._has_bos = tokenizer.bos_token_id is not None
        self._has_eos = tokenizer.eos_token_id is not None

    @property
    def num_special_tokens(self) -> int:
        return int(self._has_bos) + int(self._has_eos)

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        texts = [example[self.text_field] for example in batch]

        assert len(set(len(t) for t in texts)) == 1, (
            "All sequences must have the same length"
        )

        encodings = self._hf_tokenizer(
            texts,
            # important so input ids are aligned with loss weights
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_special_tokens_mask=False,
            return_tensors="np",
            verbose=False,
        )

        char_arrays = np.array([list(t) for t in texts], dtype="U1")
        is_upper = np.char.isupper(char_arrays)
        char_weights = np.where(
            is_upper, self.uppercase_weight, self.lowercase_weight
        ).astype(np.float32)

        input_ids = encodings["input_ids"].astype(np.int32)

        assert input_ids.shape == char_weights.shape, (
            f"Token count ({input_ids.shape[1]}) != char count ({char_weights.shape[1]}). "
            "Tokenizer must be character-level."
        )

        batch_size = input_ids.shape[0]

        # Align weights with targets: loss_weight[i] controls the loss for predicting
        # input_ids[i+1], so it should reflect the case of the *next* character.
        # Shift character weights left by 1; the last position predicts EOS (weight 1.0)
        # or is masked by not_last_mask in the loss function if there is no EOS.
        loss_weights = np.roll(char_weights, -1, axis=1)
        loss_weights[:, -1] = 1.0 if self._has_eos else 0.0

        if self._has_bos:
            bos_ids = np.full(
                (batch_size, 1), self.tokenizer.bos_token_id, dtype=np.int32
            )
            # BOS position predicts the first character — use that character's weight
            bos_weights = char_weights[:, :1]
            input_ids = np.concatenate([bos_ids, input_ids], axis=1)
            loss_weights = np.concatenate([bos_weights, loss_weights], axis=1)

        if self._has_eos:
            eos_ids = np.full(
                (batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32
            )
            eos_weights = np.ones((batch_size, 1), dtype=np.float32)
            input_ids = np.concatenate([input_ids, eos_ids], axis=1)
            loss_weights = np.concatenate([loss_weights, eos_weights], axis=1)

        return [
            {"input_ids": ids, "loss_weight": weights}
            for ids, weights in zip(input_ids, loss_weights)
        ]

    @property
    def output_exemplar(self) -> dict:
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "loss_weight": np.zeros((0,), dtype=np.float32),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "uppercase_weight": self.uppercase_weight,
            "lowercase_weight": self.lowercase_weight,
            "has_bos": self._has_bos,
            "has_eos": self._has_eos,
        }

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return min(max(1, logical_cpu_core_count() - 4), 12)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0
