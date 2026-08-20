import pytest
from marin_dna_carbon_conditioning_vep.prompts import (
    CONDITION_ORDER,
    PromptGrammar,
    assert_allele_token_count_parity,
    assert_prefix_outside_dna_mode,
    render_condition_prefix,
    render_prompt,
    truncate_dna_to_kmer_boundary,
)


class FakeCarbonTokenizer:
    dna_begin_token_id = 151669

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_token_mask: bool,
    ) -> dict[str, list[int]]:
        assert not add_special_tokens and return_token_mask
        metadata, separator, dna = text.partition("<dna>")
        assert separator == "<dna>" and dna == ""
        metadata_ids = [] if not metadata else [101, 102]
        return {
            "input_ids": metadata_ids + [self.dna_begin_token_id],
            "token_mask": [-1] * len(metadata_ids) + [0],
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        metadata, separator, dna = text.partition("<dna>")
        assert separator == "<dna>"
        assert len(dna) % 6 == 0
        metadata_ids = [] if not metadata else [101, 102]
        return metadata_ids + [self.dna_begin_token_id] + [200] * (len(dna) // 6)


def test_condition_prefixes_share_one_scaffold() -> None:
    grammar = PromptGrammar("corpus_card", "<species>{species}<dna>")
    conditions = {
        "untagged": None,
        "correct": "vertebrate_mammalian",
        "near_wrong": "vertebrate_other",
        "far_wrong": "fungi",
    }
    assert tuple(conditions) == CONDITION_ORDER
    assert render_condition_prefix("untagged", conditions, grammar) == "<dna>"
    assert (
        render_condition_prefix("correct", conditions, grammar)
        == "<species>vertebrate_mammalian<dna>"
    )
    assert (
        render_condition_prefix("near_wrong", conditions, grammar)
        == "<species>vertebrate_other<dna>"
    )
    assert (
        render_condition_prefix("far_wrong", conditions, grammar)
        == "<species>fungi<dna>"
    )


def test_prefix_stays_outside_dna_mode_and_alleles_have_equal_tokens() -> None:
    tokenizer = FakeCarbonTokenizer()
    prefix = "<species>fungi<dna>"
    assert assert_prefix_outside_dna_mode(tokenizer, prefix) == [101, 102, 151669]
    n_tokens = assert_allele_token_count_parity(
        tokenizer,
        prefix,
        "ACGTACGT",
        "ACATACGT",
    )
    assert n_tokens == 4


def test_official_six_mer_truncation() -> None:
    assert truncate_dna_to_kmer_boundary("ACGTACGT", 6) == "ACGTAC"
    assert render_prompt("<dna>", "acgtacgt", 6) == "<dna>ACGTAC"
    with pytest.raises(AssertionError, match="canonical"):
        render_prompt("<dna>", "ACGTNC", 6)


def test_invalid_condition_order_is_rejected() -> None:
    grammar = PromptGrammar("direct", "<{species}><dna>")
    with pytest.raises(AssertionError, match="condition order"):
        render_condition_prefix(
            "correct",
            {"correct": "vertebrate_mammalian", "untagged": None},
            grammar,
        )
