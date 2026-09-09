from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast

from marin_dna_exp550.formats import RagFormat, RagProcessor, RagTokenizedCache
from marin_dna_exp550.recipe import encode_document


def test_direct_encoding_matches_exported_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        Path(__file__).parents[1] / "src/marin_dna_exp550/tokenizer"
    )
    for count in (1, 3, 40):
        text = "[SEQ]".join([("AcgTN" * 51)] * count)
        direct = encode_document(text)
        observed = tokenizer(
            text, padding="max_length", max_length=10240, truncation=False
        )
        assert direct["input_ids"] == observed["input_ids"]
        assert sum(observed["attention_mask"]) == 256 * count


def test_prebuilt_cache_retains_masks_and_one_row_per_document():
    rows = [{"sequence": "A" * 255}, {"sequence": "[SEQ]".join(["C" * 255] * 2)}]
    encoded = RagProcessor()(rows)
    assert len(encoded) == 2
    assert encoded[0]["input_ids"].shape == (10240,)
    assert encoded[1]["input_ids"].shape == (10240,)
    assert encoded[0]["loss_weight"].sum() == 255
    assert encoded[1]["loss_weight"].sum() == 511
    assert encoded[0]["loss_weight"].dtype == np.float32
    assert RagTokenizedCache(path="unused").format == RagFormat()
