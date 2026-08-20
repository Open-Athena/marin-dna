# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DNA format and train-time dispatch copied into the isolated experiment."""

import inspect
from dataclasses import dataclass

from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.formats import LmDatasetFormatBase
from levanter.tokenizers import MarinTokenizer

from exp473_center_seeded_projection.batch_tokenizer import DNABatchTokenizer


@LmDatasetFormatBase.register_subclass("dna")
@dataclass(frozen=True)
class DNALmDatasetFormat(LmDatasetFormatBase):
    """DNA sequence field with case-derived per-target loss weights."""

    text_key: str = "sequence"
    uppercase_weight: float = 1.0
    lowercase_weight: float = 1.0

    def build_preprocessor(
        self,
        tokenizer: MarinTokenizer,
        *,
        enforce_eos: bool = True,
        enforce_bos: bool = True,
    ) -> BatchProcessor[dict, dict]:
        del enforce_eos, enforce_bos
        return DNABatchTokenizer(
            tokenizer,
            text_field=self.text_key,
            uppercase_weight=self.uppercase_weight,
            lowercase_weight=self.lowercase_weight,
        )


def _install_dataset_for_component_patch() -> None:
    from levanter.data.text import datasets

    if getattr(datasets.dataset_for_component, "_exp473_dna_patched", False):
        return
    original = datasets.dataset_for_component
    try:
        if "DNALmDatasetFormat" in inspect.getsource(original):
            raise RuntimeError(
                "Levanter now handles DNALmDatasetFormat upstream; remove the "
                "exp473 compatibility patch after verifying its loss-weight key"
            )
    except OSError:
        pass

    def patched(component, Pos, cache, *, eos_id, block_cross_document_attention):
        if isinstance(component.format, DNALmDatasetFormat):
            return datasets.CausalLmDataset(
                datasets.TokenSeqDataset(
                    cache,
                    Pos.size,
                    loss_weights_key="loss_weight",
                ),
                Pos,
                eos_id=eos_id,
                block_cross_document_attention=block_cross_document_attention,
            )
        return original(
            component,
            Pos,
            cache,
            eos_id=eos_id,
            block_cross_document_attention=block_cross_document_attention,
        )

    patched._exp473_dna_patched = True  # type: ignore[attr-defined]
    datasets.dataset_for_component = patched


_install_dataset_for_component_patch()
