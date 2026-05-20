# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``marin_dna.levanter`` glue (format registration + seq-len helper).

These cover behaviors that are marin-dna-specific (i.e. not just ports of
upstream marin tests).
"""

import pytest

pytest.importorskip("levanter", reason="install with `uv sync --extra marin` to run")

from levanter.data.text.formats import LmDatasetFormatBase  # noqa: E402

from marin_dna.levanter import formats as _formats_module  # noqa: E402, F401  side-effect: register "dna"
from marin_dna.levanter.batch_tokenizer import DNABatchTokenizer  # noqa: E402
from marin_dna.levanter.defaults import dna_effective_seq_len  # noqa: E402
from marin_dna.levanter.formats import DNALmDatasetFormat  # noqa: E402


BOS_EOS_TOKENIZER = "bolinas-dna/tokenizer-char"
NO_BOS_EOS_TOKENIZER = "songlab/tokenizer-dna-clm"


def _tokenizer_available(name: str) -> bool:
    from levanter.tokenizers import load_tokenizer

    try:
        load_tokenizer(name)
    except Exception:
        return False
    return True


def test_dna_format_registered_on_import():
    """Importing ``marin_dna.levanter.formats`` activates ``register_subclass('dna')``."""
    choices = LmDatasetFormatBase.get_known_choices()
    assert "dna" in choices, f"'dna' not registered; got {sorted(choices)}"


def test_dna_format_round_trips_via_choice_registry():
    """The registered class must round-trip through ChoiceRegistry by key."""
    cls = LmDatasetFormatBase.get_choice_class("dna")
    assert cls is DNALmDatasetFormat


@pytest.mark.skipif(
    not _tokenizer_available(BOS_EOS_TOKENIZER),
    reason=f"Tokenizer {BOS_EOS_TOKENIZER} not accessible",
)
def test_dna_effective_seq_len_with_bos_eos():
    """255 bp base + BOS + EOS = 257 tokens."""
    assert dna_effective_seq_len(255, BOS_EOS_TOKENIZER) == 257


@pytest.mark.skipif(
    not _tokenizer_available(NO_BOS_EOS_TOKENIZER),
    reason=f"Tokenizer {NO_BOS_EOS_TOKENIZER} not accessible",
)
def test_dna_effective_seq_len_with_no_special_tokens():
    """No BOS/EOS → effective length matches base."""
    assert dna_effective_seq_len(255, NO_BOS_EOS_TOKENIZER) == 255


def test_dna_format_build_preprocessor_returns_dna_batch_tokenizer():
    """Format's ``build_preprocessor`` must return a ``DNABatchTokenizer`` instance.

    Uses a stub tokenizer so the test doesn't hit HF.
    """

    class _StubTokenizer:
        bos_token_id = None
        eos_token_id = None
        name_or_path = "stub"
        vocab_size = 0

        def as_hf_tokenizer(self):
            return None

    fmt = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)
    proc = fmt.build_preprocessor(_StubTokenizer())
    assert isinstance(proc, DNABatchTokenizer)
    assert proc.text_field == "sequence"
    assert proc.lowercase_weight == 0.01
    assert proc.uppercase_weight == 1.0
