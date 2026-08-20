from typing import Any

import pytest
from marin_dna_carbon_conditioning_vep.tokenizer_snapshot import (
    FROZEN_PREFIX_TOKEN_IDS,
    assert_frozen_prefix_token_ids,
)

GRAMMARS = {
    "model_card": "<{species}><dna>",
    "corpus_card": "<species>{species}<dna>",
}
CONDITIONS = {
    "untagged": None,
    "correct": "vertebrate_mammalian",
    "near_wrong": "vertebrate_other",
    "far_wrong": "fungi",
}


class SnapshotTokenizer:
    dna_begin_token_id = 151669

    def __init__(self, *, corrupt: bool = False) -> None:
        self.corrupt = corrupt

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_token_mask: bool,
    ) -> dict[str, Any]:
        assert not add_special_tokens and return_token_mask
        for grammar_name, template in GRAMMARS.items():
            for condition, species in CONDITIONS.items():
                prefix = (
                    "<dna>" if species is None else template.format(species=species)
                )
                if prefix != text:
                    continue
                ids = list(FROZEN_PREFIX_TOKEN_IDS[grammar_name][condition])
                if (
                    self.corrupt
                    and grammar_name == "model_card"
                    and condition == "correct"
                ):
                    ids[0] += 1
                return {
                    "input_ids": ids,
                    "token_mask": [-1] * (len(ids) - 1) + [0],
                }
        raise AssertionError(f"unexpected prefix {text!r}")


def test_pinned_carbon_prefix_token_ids_match_snapshot() -> None:
    assert (
        assert_frozen_prefix_token_ids(SnapshotTokenizer(), GRAMMARS, CONDITIONS)
        == FROZEN_PREFIX_TOKEN_IDS
    )


def test_tokenizer_snapshot_drift_blocks_inference() -> None:
    with pytest.raises(AssertionError, match="tokenizer IDs changed"):
        assert_frozen_prefix_token_ids(
            SnapshotTokenizer(corrupt=True),
            GRAMMARS,
            CONDITIONS,
        )
