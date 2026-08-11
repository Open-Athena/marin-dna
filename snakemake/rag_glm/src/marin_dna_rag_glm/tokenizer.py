"""Tokenizer factory for the fixed-layout issue #402 documents."""

from transformers import PreTrainedTokenizerFast

from marin_dna_rag_glm.char_tokenizer import (
    SEQUENCE_BOUNDARY_TOKEN,
    create_char_tokenizer,
)


def create_rag_char_tokenizer() -> PreTrainedTokenizerFast:
    """Create the BOS-as-CLS, no-EOS tokenizer used by the RAG prototype."""
    tokenizer = create_char_tokenizer(
        bos=True,
        eos=False,
        sequence_boundary=True,
    )
    assert tokenizer.cls_token_id == tokenizer.bos_token_id
    assert tokenizer.eos_token_id is None
    boundary_id = tokenizer.convert_tokens_to_ids(SEQUENCE_BOUNDARY_TOKEN)
    assert boundary_id != tokenizer.unk_token_id
    assert boundary_id < tokenizer.vocab_size
    return tokenizer
