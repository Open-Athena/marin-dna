"""Tests for the issue #402 tokenizer publication card."""

from scripts.issue402_upload_rag_tokenizer import tokenizer_readme


def test_tokenizer_card_has_required_tags_and_pinned_provenance() -> None:
    sha = "a" * 40
    readme = tokenizer_readme(sha)
    assert "- biology" in readme
    assert "- genomics" in readme
    assert "- dna" in readme
    assert f"blob/{sha}/scripts/issue402_upload_rag_tokenizer.py" in readme
    assert "[BOS]" in readme
    assert "[SEQ]" in readme
