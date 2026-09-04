from __future__ import annotations

import numpy as np
from exp517_functional_specialists.batch_tokenizer import DNABatchTokenizer
from exp517_functional_specialists.formats import DNALmDatasetFormat
from levanter.data.text import datasets
from levanter.data.text.formats import LmDatasetFormatBase


class _StubHfTokenizer:
    def __call__(self, texts, **_kwargs):
        values = {base: index for index, base in enumerate("ACGTacgt", start=1)}
        return {
            "input_ids": np.asarray(
                [[values[base] for base in text] for text in texts],
                dtype=np.int32,
            )
        }


class _StubTokenizer:
    bos_token_id = 9
    eos_token_id = None
    name_or_path = "stub-char-bos"
    vocab_size = 10

    def as_hf_tokenizer(self):
        return _StubHfTokenizer()


def test_dna_format_and_train_dispatch_are_registered() -> None:
    assert LmDatasetFormatBase.get_choice_class("dna") is DNALmDatasetFormat
    assert getattr(datasets.dataset_for_component, "_exp517_dna_patched", False)


def test_case_weights_are_target_aligned_with_bos() -> None:
    tokenizer = DNABatchTokenizer(
        _StubTokenizer(),
        text_field="sequence",
        lowercase_weight=0.1,
    )
    result = tokenizer([{"sequence": "ACgt"}])[0]
    assert result["input_ids"].shape == (5,)
    assert result["input_ids"][0] == 9
    np.testing.assert_allclose(
        result["loss_weight"],
        np.asarray([1.0, 1.0, 0.1, 0.1, 0.0], dtype=np.float32),
    )


def test_dna_dispatch_reads_singular_loss_weight_key(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubTokenSeq:
        def __init__(self, _cache, seq_len, loss_weights_key=None):
            captured["seq_len"] = seq_len
            captured["loss_weights_key"] = loss_weights_key

    class StubCausal:
        def __init__(self, inner, _pos, *, eos_id, block_cross_document_attention):
            captured["inner"] = inner
            captured["eos_id"] = eos_id
            captured["block"] = block_cross_document_attention

    class Pos:
        size = 256

    class Component:
        format = DNALmDatasetFormat()

    monkeypatch.setattr(datasets, "TokenSeqDataset", StubTokenSeq)
    monkeypatch.setattr(datasets, "CausalLmDataset", StubCausal)
    result = datasets.dataset_for_component(
        Component(),
        Pos(),
        cache=object(),
        eos_id=7,
        block_cross_document_attention=True,
    )
    assert isinstance(result, StubCausal)
    assert captured == {
        "seq_len": 256,
        "loss_weights_key": "loss_weight",
        "inner": captured["inner"],
        "eos_id": 7,
        "block": True,
    }
