"""Tests for the DNA k-mer tokenizer (HuggingFace API)."""

import itertools
import tempfile

import pytest
from transformers import AutoTokenizer

from marin_dna.tokenizer.kmer import SPECIAL_TOKENS, create_kmer_tokenizer

# ---------------------------------------------------------------------------
# Vocab sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, 6), (2, 18), (3, 66), (6, 4098)],
    ids=["k1", "k2", "k3", "k6"],
)
def test_vocab_size(k, expected):
    tok = create_kmer_tokenizer(k)
    assert tok.vocab_size == expected


def test_special_token_ids():
    tok = create_kmer_tokenizer(k=1)
    assert tok.convert_tokens_to_ids("[PAD]") == 0
    assert tok.convert_tokens_to_ids("[UNK]") == 1


def test_k1_ordering():
    tok = create_kmer_tokenizer(k=1)
    assert tok.convert_tokens_to_ids("a") == 2
    assert tok.convert_tokens_to_ids("c") == 3
    assert tok.convert_tokens_to_ids("g") == 4
    assert tok.convert_tokens_to_ids("t") == 5


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


def test_call_returns_batch_encoding():
    tok = create_kmer_tokenizer(k=3)
    out = tok("acgttt")
    assert "input_ids" in out
    assert "attention_mask" in out
    assert len(out["input_ids"]) == 2
    assert out["attention_mask"] == [1, 1]


def test_call_batch():
    tok = create_kmer_tokenizer(k=3)
    out = tok(["acgttt", "gggaaa"])
    assert len(out["input_ids"]) == 2
    assert len(out["input_ids"][0]) == 2
    assert len(out["input_ids"][1]) == 2


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def test_encode_single_kmer():
    tok = create_kmer_tokenizer(k=3)
    ids = tok.encode("acg")
    assert ids == [tok.convert_tokens_to_ids("acg")]


def test_encode_multiple_kmers():
    tok = create_kmer_tokenizer(k=3)
    ids = tok.encode("acgttt")
    expected = [tok.convert_tokens_to_ids("acg"), tok.convert_tokens_to_ids("ttt")]
    assert ids == expected


def test_encode_k1():
    tok = create_kmer_tokenizer(k=1)
    ids = tok.encode("acgt")
    assert ids == [2, 3, 4, 5]


def test_encode_k2():
    tok = create_kmer_tokenizer(k=2)
    ids = tok.encode("acgt")
    assert ids == [tok.convert_tokens_to_ids("ac"), tok.convert_tokens_to_ids("gt")]


def test_encode_case_insensitive():
    tok = create_kmer_tokenizer(k=3)
    lower = tok.encode("acgttt")
    upper = tok.encode("ACGTTT")
    mixed = tok.encode("AcGtTt")
    assert lower == upper == mixed


def test_encode_empty():
    tok = create_kmer_tokenizer(k=3)
    assert tok.encode("") == []


def test_encode_n_maps_to_unk():
    tok = create_kmer_tokenizer(k=3)
    ids = tok.encode("ang")
    assert ids == [1]  # [UNK]


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def test_decode_roundtrip():
    tok = create_kmer_tokenizer(k=3)
    seq = "acgtttggg"
    assert tok.decode(tok.encode(seq)) == seq


def test_decode_skip_special_tokens():
    tok = create_kmer_tokenizer(k=3)
    pad_id = tok.convert_tokens_to_ids("[PAD]")
    acg_id = tok.convert_tokens_to_ids("acg")
    assert tok.decode([pad_id, acg_id], skip_special_tokens=True) == "acg"


def test_decode_empty():
    tok = create_kmer_tokenizer(k=3)
    assert tok.decode([]) == ""


# ---------------------------------------------------------------------------
# Lookup: convert_tokens_to_ids / convert_ids_to_tokens
# ---------------------------------------------------------------------------


def test_convert_tokens_to_ids():
    tok = create_kmer_tokenizer(k=3)
    assert isinstance(tok.convert_tokens_to_ids("acg"), int)


def test_convert_ids_to_tokens():
    tok = create_kmer_tokenizer(k=3)
    token = tok.convert_ids_to_tokens(2)
    assert token == "aaa"  # first k-mer in lex order


def test_convert_special_tokens():
    tok = create_kmer_tokenizer(k=3)
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.convert_tokens_to_ids(name) == i
        assert tok.convert_ids_to_tokens(i) == name


# ---------------------------------------------------------------------------
# Roundtrip / uniqueness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_encode_decode_roundtrip(k):
    tok = create_kmer_tokenizer(k)
    seq = "acgt" * k  # length 4k, always a multiple of k
    assert tok.decode(tok.encode(seq)) == seq


def test_all_kmers_unique_ids():
    tok = create_kmer_tokenizer(k=3)
    all_ids = set()
    for bases in itertools.product("acgt", repeat=3):
        kmer = "".join(bases)
        token_id = tok.convert_tokens_to_ids(kmer)
        assert token_id is not None
        assert token_id not in all_ids
        all_ids.add(token_id)
    assert len(all_ids) == 64


# ---------------------------------------------------------------------------
# save_pretrained / AutoTokenizer.from_pretrained
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip():
    tok = create_kmer_tokenizer(k=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save_pretrained(tmpdir)
        loaded = AutoTokenizer.from_pretrained(tmpdir)

        seq = "acgtttggg"
        assert loaded.encode(seq) == tok.encode(seq)
        assert loaded.decode(loaded.encode(seq)) == seq
