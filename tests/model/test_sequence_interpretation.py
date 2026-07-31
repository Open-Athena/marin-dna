"""Regression tests for the issue #387 single-sequence logo definition."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from marin_dna.data.dna import reverse_complement
from marin_dna.model.interpretation import nucleotide_dependency_map
from marin_dna.model.sequence_interpretation import (
    _RC_CHANNEL_INDICES,
    _strand_nucleotide_logits,
    interpret_sequence,
    normalize_dna_sequence,
    nucleotide_logo,
)
from marin_dna.tokenizer.char import create_char_tokenizer


class _PrefixSumCausalLM(nn.Module):
    """Small causal, content-dependent test double with non-trivial logits."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        x = input_ids.float()
        prefix = torch.cumsum(x, dim=1)
        positions = torch.arange(input_ids.shape[1], dtype=torch.float)
        vocabulary = torch.arange(self.vocab_size, dtype=torch.float)
        logits = torch.sin(
            0.31 * prefix.unsqueeze(-1)
            + 0.17 * positions.view(1, -1, 1)
            + 0.53 * vocabulary
        )
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" acgt acgt\nACGT\tACGT ", "ACGTACGTACGTACGT"),
        ("a" * 16, "A" * 16),
    ],
)
def test_normalize_dna_sequence(raw, expected):
    assert normalize_dna_sequence(raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "empty"),
        (" \n\t", "empty"),
        ("ACGTN" * 4, "unsupported"),
        ("ACGT!" * 4, "unsupported"),
        ("ACGT", "too short"),
        ("A" * 256, "too long"),
    ],
)
def test_normalize_dna_sequence_rejects_invalid_input(raw, message):
    with pytest.raises(ValueError, match=message):
        normalize_dna_sequence(raw)


def test_nucleotide_logo_averages_logits_before_softmax():
    tokenizer = create_char_tokenizer(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    sequence = "ACGTACGTACGTACGT"

    forward = _strand_nucleotide_logits(model, tokenizer, sequence)
    reverse = _strand_nucleotide_logits(model, tokenizer, reverse_complement(sequence))
    transformed_reverse = reverse.flip(0)[:, _RC_CHANNEL_INDICES]
    expected = torch.softmax((forward + transformed_reverse) / 2.0, dim=-1).numpy()
    probability_mean = (
        torch.softmax(forward, dim=-1) + torch.softmax(transformed_reverse, dim=-1)
    ).numpy() / 2.0

    actual = nucleotide_logo(model, tokenizer, sequence)
    np.testing.assert_allclose(actual.probabilities, expected, atol=1e-7)
    assert not np.allclose(actual.probabilities, probability_mean)


def test_nucleotide_logo_invariants():
    tokenizer = create_char_tokenizer(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    sequence = "AAAACCCCGGGGTTTT"
    logo = nucleotide_logo(model, tokenizer, sequence)

    assert logo.probabilities.shape == (len(sequence), 4)
    assert logo.entropy_bits.shape == (len(sequence),)
    assert logo.information_bits.shape == (len(sequence),)
    assert logo.glyph_heights_bits.shape == (len(sequence), 4)
    assert np.isfinite(logo.probabilities).all()
    assert (logo.probabilities >= 0).all()
    np.testing.assert_allclose(logo.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert ((0.0 <= logo.entropy_bits) & (logo.entropy_bits <= 2.0)).all()
    assert ((0.0 <= logo.information_bits) & (logo.information_bits <= 2.0)).all()
    np.testing.assert_allclose(
        logo.glyph_heights_bits.sum(axis=1),
        logo.information_bits,
        atol=1e-6,
    )


def test_nucleotide_logo_forward_reverse_complement_equivariance():
    tokenizer = create_char_tokenizer(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    sequence = "AAAACGTCGATCTTGC"
    forward = nucleotide_logo(model, tokenizer, sequence)
    reverse = nucleotide_logo(model, tokenizer, reverse_complement(sequence))

    reverse_probabilities = reverse.probabilities[::-1][:, _RC_CHANNEL_INDICES]
    reverse_heights = reverse.glyph_heights_bits[::-1][:, _RC_CHANNEL_INDICES]
    np.testing.assert_allclose(forward.probabilities, reverse_probabilities, atol=1e-7)
    np.testing.assert_allclose(forward.glyph_heights_bits, reverse_heights, atol=1e-7)
    np.testing.assert_allclose(
        forward.entropy_bits, reverse.entropy_bits[::-1], atol=1e-7
    )


def test_nucleotide_logo_requires_bos():
    tokenizer = create_char_tokenizer(bos=False, eos=False)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    with pytest.raises(AssertionError, match="BOS"):
        nucleotide_logo(model, tokenizer, "ACGTACGTACGTACGT")


def test_interpret_sequence_matches_dependency_implementation():
    tokenizer = create_char_tokenizer(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    raw = " acgt acgt acgt acgt "
    result = interpret_sequence(model, tokenizer, raw, batch_size=7)
    expected_dependency = nucleotide_dependency_map(
        model,
        tokenizer,
        result.sequence,
        rc=True,
        combine="mean",
        batch_size=7,
    )

    assert result.sequence == "ACGTACGTACGTACGT"
    np.testing.assert_allclose(result.dependency, expected_dependency)
    assert np.allclose(result.dependency, result.dependency.T)
