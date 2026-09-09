"""One published RAG row becomes one fixed-shape, masked training example."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.formats import LmDatasetFormatBase, PrebuiltLmDatasetFormat
from marin.processing.tokenize.tokenize import TokenizedCache

from marin_dna_exp550.recipe import MODEL_TOKENS, encode_document


class RagProcessor(BatchProcessor[dict, dict]):
    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        result = []
        for row in batch:
            encoded = encode_document(row["sequence"])
            result.append(
                {
                    "input_ids": np.asarray(encoded["input_ids"], dtype=np.int32),
                    "loss_weight": np.asarray(encoded["loss_weight"], dtype=np.float32),
                }
            )
        return result

    @property
    def output_exemplar(self) -> dict[str, np.ndarray]:
        return {
            "input_ids": np.zeros(0, dtype=np.int32),
            "loss_weight": np.zeros(0, dtype=np.float32),
        }

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> dict[str, Any]:
        return {"contract": "rag-255-bos-separator-rightpad-v1", "length": MODEL_TOKENS}


@LmDatasetFormatBase.register_subclass("exp550_rag")
@dataclass(frozen=True)
class RagFormat(PrebuiltLmDatasetFormat):
    loss_weights_key: str | None = "loss_weight"

    def build_preprocessor(
        self, tokenizer: Any, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> RagProcessor:
        del tokenizer, enforce_eos, enforce_bos
        return RagProcessor()


class RagTokenizedCache(TokenizedCache):
    @property
    def format(self) -> RagFormat:
        # The published Marin base artifact currently reconstructs text format.
        # Retain the supported Prebuilt dataset path without patching global APIs.
        return RagFormat()
