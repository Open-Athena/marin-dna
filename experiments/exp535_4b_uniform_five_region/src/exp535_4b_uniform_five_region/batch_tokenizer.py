# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Case-aware DNA tokenizer vendored into the isolated issue #535 project."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from levanter.data._preprocessor import BatchProcessor
from levanter.tokenizers import MarinTokenizer
from levanter.utils.py_utils import logical_cpu_core_count


class DNABatchTokenizer(BatchProcessor[dict, dict]):
    """Tokenize fixed-length DNA and target-align loss weights by letter case."""

    def __init__(
        self,
        tokenizer: MarinTokenizer,
        text_field: str = "sequence",
        uppercase_weight: float = 1.0,
        lowercase_weight: float = 1.0,
        *,
        override_resources: dict[str, Any] | None = None,
    ) -> None:
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
        assert len({len(text) for text in texts}) == 1, "all sequences must have the same length"
        encodings = self._hf_tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_special_tokens_mask=False,
            return_tensors="np",
            verbose=False,
        )
        char_arrays = np.array([list(text) for text in texts], dtype="U1")
        char_weights = np.where(
            np.char.isupper(char_arrays),
            self.uppercase_weight,
            self.lowercase_weight,
        ).astype(np.float32)
        input_ids = encodings["input_ids"].astype(np.int32)
        assert input_ids.shape == char_weights.shape, (
            f"token count {input_ids.shape[1]} != character count {char_weights.shape[1]}; "
            "the tokenizer must be character-level"
        )
        batch_size = input_ids.shape[0]
        loss_weights = np.roll(char_weights, -1, axis=1)
        loss_weights[:, -1] = 1.0 if self._has_eos else 0.0
        if self._has_bos:
            bos_ids = np.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=np.int32)
            input_ids = np.concatenate([bos_ids, input_ids], axis=1)
            loss_weights = np.concatenate([char_weights[:, :1], loss_weights], axis=1)
        if self._has_eos:
            eos_ids = np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)
            input_ids = np.concatenate([input_ids, eos_ids], axis=1)
            loss_weights = np.concatenate(
                [loss_weights, np.ones((batch_size, 1), dtype=np.float32)],
                axis=1,
            )
        return [
            {"input_ids": ids, "loss_weight": weights}
            for ids, weights in zip(input_ids, loss_weights, strict=True)
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
            cpus = self.override_resources.get("num_cpus")
            if cpus is not None:
                return int(cpus)
        return min(max(1, logical_cpu_core_count() - 4), 12)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return int(self.override_resources.get("num_gpus", 0))
        return 0
