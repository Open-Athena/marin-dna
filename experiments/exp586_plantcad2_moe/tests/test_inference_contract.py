import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from exp586_moe.inference.contract import token_contract, validate_sequences


def character_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "a": 1, "c": 2, "g": 3, "t": 4, "n": 5})
    )
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return tokenizer


def test_token_contract_preserves_character_ids() -> None:
    contract = token_contract(character_tokenizer())
    assert contract.input_ids == {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
    assert contract.output_ids == {"A": 1, "C": 2, "G": 3, "T": 4}


def test_sequences_are_unique_complete_windows() -> None:
    validate_sequences(["ACGTN" * 1_638 + "AC"])
    with pytest.raises(ValueError, match="unique"):
        validate_sequences(["A" * 8192, "A" * 8192])
    with pytest.raises(ValueError, match="outside ACGTN"):
        validate_sequences(["X" * 8192])
