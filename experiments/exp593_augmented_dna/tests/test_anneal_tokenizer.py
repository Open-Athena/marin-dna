import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from exp593_dna import anneal_tokenizer
from exp593_dna.pipeline import ALPHABET, Encoder


def tokenizers(monkeypatch):
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}}
    text = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    text.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    source = Tokenizer.from_str(text.to_str())
    source.normalizer = normalizers.Lowercase()
    source.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    monkeypatch.setattr(anneal_tokenizer, "pretrained_dna_tokenizer", lambda: source)
    monkeypatch.setattr(
        anneal_tokenizer,
        "SOURCE_TOKENIZER_SHA256",
        anneal_tokenizer.tokenizer_digest(source),
    )
    return Encoder(text), source


def test_distinct_json_hashes_preserve_all_vocabulary_and_dna_ids(monkeypatch):
    encoder, _ = tokenizers(monkeypatch)
    evidence = anneal_tokenizer.verify_source_tokenizer(encoder)
    assert evidence["source_tokenizer_sha256"] != evidence["text_tokenizer_sha256"]
    assert evidence["dna_contract"] == "identical_lowercase_character_ids"


def test_changed_dna_character_mapping_is_rejected(monkeypatch):
    encoder, _ = tokenizers(monkeypatch)
    encoder.dna_lookup[ord("c")] = encoder.dna_lookup[ord("a")]
    with pytest.raises(ValueError, match="character IDs differ"):
        anneal_tokenizer.verify_source_tokenizer(encoder)


def test_changed_vocabulary_and_source_identity_are_rejected(monkeypatch):
    encoder, _ = tokenizers(monkeypatch)
    encoder.tokenizer.add_tokens(["unexpected-extra-token"])
    with pytest.raises(ValueError, match="vocabulary IDs differ"):
        anneal_tokenizer.verify_source_tokenizer(encoder)
    encoder, _ = tokenizers(monkeypatch)
    monkeypatch.setattr(anneal_tokenizer, "SOURCE_TOKENIZER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="tokenizer has changed"):
        anneal_tokenizer.verify_source_tokenizer(encoder)
