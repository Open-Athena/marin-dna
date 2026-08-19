from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from exp479_mntp.vep import LoadedArm, assert_development_split, reverse_complement, score_strand


class CharacterTokenizer:
    mask_token_id = 7

    def __call__(
        self,
        sequences: str | list[str],
        *,
        return_tensors: str,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        assert return_tensors == "pt"
        if isinstance(sequences, str):
            sequences = [sequences]
        lookup = {"A": 3, "C": 4, "G": 5, "T": 6}
        input_ids = torch.tensor([[2, *(lookup[base] for base in seq)] for seq in sequences])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _arm() -> LoadedArm:
    torch.manual_seed(479)
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=300,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    return LoadedArm(
        model=model,
        tokenizer=CharacterTokenizer(),
        canonical_ids=(3, 4, 5, 6),
        mask_token_id=7,
    )


def test_development_split_rejects_even_autosome() -> None:
    assert_development_split(pd.DataFrame({"chrom": ["1", "X"], "label": [0, 1]}), "ok")
    with pytest.raises(RuntimeError, match="held-out"):
        assert_development_split(pd.DataFrame({"chrom": ["2"], "label": [1]}), "bad")


def test_reverse_complement_preserves_unknowns_and_case() -> None:
    assert reverse_complement("ACGTNacgtn") == "nacgtNACGT"


def test_mntp_score_cannot_see_true_base_under_mask() -> None:
    first = "A" * 255
    second = first[:127] + "G" + first[128:]
    frame = pd.DataFrame(
        {
            "sequence": [first, second],
            "ref": ["A", "A"],
            "alt": ["C", "C"],
        }
    )
    scores = score_strand(_arm(), frame, objective="mntp", strand="fwd", batch_size=2)
    assert scores[0] == scores[1]
