"""Levanter batch tokenizer for issue #402 fixed-layout RAG documents."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from levanter.tokenizers import MarinTokenizer

from marin_dna_evals.levanter.batch_tokenizer import DNABatchTokenizer
from marin_dna.pipelines.rag_glm.dataset import (
    BASES_PER_SLOT,
    DOCUMENT_TOKENS,
    DOCUMENT_TOKENS_WITHOUT_CLS,
    N_NON_HUMAN_SLOTS,
    SEQUENCE_BOUNDARY,
)


class RAGDNABatchTokenizer(DNABatchTokenizer):
    """Tokenize eight-slot documents with atomic boundaries and uniform loss.

    The input text contains 2,040 bases plus seven literal ``[SEQ]`` markers.
    Tokenization must yield 2,047 tokens before BOS/CLS and 2,048 afterward.
    Every defined causal target receives weight 1; only the final position,
    which has no next-token target, receives weight 0.
    """

    def __init__(
        self,
        tokenizer: MarinTokenizer,
        text_field: str = "seq",
        *,
        override_resources: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            tokenizer,
            text_field=text_field,
            uppercase_weight=1.0,
            lowercase_weight=1.0,
            override_resources=override_resources,
        )
        assert self._has_bos, "RAG documents require one BOS/CLS token"
        assert not self._has_eos, "RAG documents are exactly 2,048 tokens without EOS"
        assert self._hf_tokenizer.cls_token_id == tokenizer.bos_token_id
        self._sequence_boundary_id = self._hf_tokenizer.convert_tokens_to_ids(
            SEQUENCE_BOUNDARY
        )
        assert self._sequence_boundary_id != self._hf_tokenizer.unk_token_id

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        texts = [example[self.text_field] for example in batch]
        assert texts, "batch is empty"
        assert all(text.count(SEQUENCE_BOUNDARY) == N_NON_HUMAN_SLOTS for text in texts)

        encodings = self._hf_tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_special_tokens_mask=False,
            return_tensors="np",
            verbose=False,
        )
        input_ids = encodings["input_ids"].astype(np.int32)
        assert input_ids.shape == (len(texts), DOCUMENT_TOKENS_WITHOUT_CLS), (
            f"expected {DOCUMENT_TOKENS_WITHOUT_CLS} pre-CLS tokens, "
            f"got {input_ids.shape}"
        )

        bos_ids = np.full((len(texts), 1), self.tokenizer.bos_token_id, dtype=np.int32)
        input_ids = np.concatenate([bos_ids, input_ids], axis=1)
        assert input_ids.shape == (len(texts), DOCUMENT_TOKENS)

        boundary_positions = [
            (slot + 1) * (BASES_PER_SLOT + 1) for slot in range(N_NON_HUMAN_SLOTS)
        ]
        assert np.all(input_ids[:, boundary_positions] == self._sequence_boundary_id), (
            "literal [SEQ] was not encoded as one atomic token at every boundary"
        )

        loss_weights = np.ones(input_ids.shape, dtype=np.float32)
        loss_weights[:, -1] = 0.0
        assert np.all(loss_weights[:, :-1] == 1.0)
        assert np.all(loss_weights[:, -1] == 0.0)
        return [
            {"input_ids": ids, "loss_weight": weights}
            for ids, weights in zip(input_ids, loss_weights)
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "has_bos": self._has_bos,
            "has_eos": self._has_eos,
            "sequence_boundary_id": self._sequence_boundary_id,
            "document_tokens": DOCUMENT_TOKENS,
            "uniform_loss_weight": True,
        }
