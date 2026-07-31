"""Sequence-direct model interpretation for the public sequence explorer.

The probability-logo computation lives here; the categorical-Jacobian
calculation remains in :mod:`marin_dna.model.interpretation`.

This module is retained on the issue #419 experimental branch so the
batch-oriented genome scorer can be checked directly against the exact
single-sequence implementation developed for issue #387.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from marin_dna.data.dna import NUCLEOTIDES, reverse_complement
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
)
from marin_dna.model.interpretation import nucleotide_dependency_map

DEFAULT_MIN_SEQUENCE_LENGTH = 16
DEFAULT_MAX_SEQUENCE_LENGTH = 255
_RC_CHANNEL_INDICES = [3, 2, 1, 0]  # A<-T, C<-G, G<-C, T<-A


@dataclass(frozen=True)
class NucleotideLogo:
    """Arrays defining the logo, with matrix columns in A/C/G/T order."""

    probabilities: np.ndarray
    entropy_bits: np.ndarray
    information_bits: np.ndarray
    glyph_heights_bits: np.ndarray


@dataclass(frozen=True)
class SequenceInterpretation:
    """Complete sequence-direct interpretation."""

    sequence: str
    logo: NucleotideLogo
    dependency: np.ndarray


def normalize_dna_sequence(
    sequence: str,
    *,
    min_length: int = DEFAULT_MIN_SEQUENCE_LENGTH,
    max_length: int | None = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> str:
    """Remove whitespace, uppercase, and validate an A/C/G/T sequence."""
    assert min_length >= 1, f"min_length must be positive, got {min_length}"
    assert max_length is None or max_length >= min_length, (
        f"max_length {max_length} must be >= min_length {min_length}"
    )
    if not isinstance(sequence, str):
        raise ValueError("Sequence must be text containing only A, C, G, and T.")

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        maximum = max_length if max_length is not None else "more"
        raise ValueError(
            f"Sequence is empty. Enter between {min_length} and {maximum} "
            "A/C/G/T bases."
        )
    invalid = sorted(set(normalized) - set(NUCLEOTIDES))
    if invalid:
        display = ", ".join(repr(char) for char in invalid)
        raise ValueError(
            "Sequence contains unsupported characters. Only A, C, G, and T are "
            f"accepted; found: {display}."
        )
    if len(normalized) < min_length:
        raise ValueError(
            f"Sequence is too short ({len(normalized)} bp). Minimum length is "
            f"{min_length} bp."
        )
    if max_length is not None and len(normalized) > max_length:
        raise ValueError(
            f"Sequence is too long ({len(normalized)} bp). Maximum length is "
            f"{max_length} bp."
        )
    return normalized


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def _strand_nucleotide_logits(
    model: Any,
    tokenizer: Any,
    sequence: str,
) -> Tensor:
    """Return A/C/G/T next-token logits for every position on one strand."""
    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_prefix >= 1, (
        "the sequence-logo calculation requires an auto-prepended BOS token so "
        "the causal model predicts every sequence position"
    )

    token_ids_by_nucleotide = _get_nucleotide_token_ids(tokenizer)
    nucleotide_token_ids = torch.tensor(
        [token_ids_by_nucleotide[nuc] for nuc in NUCLEOTIDES],
        dtype=torch.long,
        device=_model_device(model),
    )
    input_ids = torch.tensor(
        tokenizer.encode(sequence),
        dtype=torch.long,
        device=nucleotide_token_ids.device,
    )
    expected_tokens = len(sequence) + n_prefix + n_suffix
    assert input_ids.shape == (expected_tokens,), (
        "sequence explorer requires one token per nucleotide: encoded "
        f"{len(sequence)} bases as {int(input_ids.shape[0])} tokens, expected "
        f"{expected_tokens} including {n_prefix} prefix and {n_suffix} suffix tokens"
    )

    logits = model(input_ids.unsqueeze(0)).logits
    assert logits.ndim == 3 and logits.shape[0] == 1, (
        f"expected model logits [1,L,V], got {tuple(logits.shape)}"
    )
    readout_indices = (
        torch.arange(len(sequence), device=input_ids.device) + n_prefix - 1
    )
    nucleotide_logits = logits[0, readout_indices][..., nucleotide_token_ids].float()
    assert nucleotide_logits.shape == (len(sequence), len(NUCLEOTIDES))
    assert torch.isfinite(nucleotide_logits).all(), (
        "model produced non-finite nucleotide logits"
    )
    return nucleotide_logits


@torch.no_grad()
def nucleotide_logo(
    model: Any,
    tokenizer: Any,
    sequence: str,
) -> NucleotideLogo:
    """Compute the forward/reverse-complement information-content logo."""
    sequence = normalize_dna_sequence(sequence, min_length=1, max_length=None)
    forward_logits = _strand_nucleotide_logits(model, tokenizer, sequence)
    reverse_logits = _strand_nucleotide_logits(
        model, tokenizer, reverse_complement(sequence)
    )
    reverse_in_forward_coordinates = reverse_logits.flip(0)[:, _RC_CHANNEL_INDICES]
    mean_logits = (forward_logits + reverse_in_forward_coordinates) / 2.0
    probabilities = torch.softmax(mean_logits, dim=-1).cpu().numpy()

    log_probabilities = np.zeros_like(probabilities)
    np.log2(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy_bits = -(probabilities * log_probabilities).sum(axis=1)
    entropy_bits = np.clip(entropy_bits, 0.0, 2.0)
    information_bits = 2.0 - entropy_bits
    glyph_heights_bits = probabilities * information_bits[:, None]

    assert np.isfinite(probabilities).all(), "logo probabilities are non-finite"
    assert (probabilities >= 0).all(), "logo probabilities must be non-negative"
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6), (
        "logo probabilities do not sum to one"
    )
    assert ((0.0 <= entropy_bits) & (entropy_bits <= 2.0)).all()
    assert ((0.0 <= information_bits) & (information_bits <= 2.0)).all()
    assert np.allclose(glyph_heights_bits.sum(axis=1), information_bits, atol=1e-6)

    return NucleotideLogo(
        probabilities=probabilities,
        entropy_bits=entropy_bits,
        information_bits=information_bits,
        glyph_heights_bits=glyph_heights_bits,
    )


def interpret_sequence(
    model: Any,
    tokenizer: Any,
    sequence: str,
    *,
    combine: Literal["mean", "max"] = "mean",
    norm_ord: float = np.inf,
    batch_size: int = 32,
    atol: float = 1e-3,
) -> SequenceInterpretation:
    """Compute both requested views directly from one validated DNA sequence."""
    normalized = normalize_dna_sequence(sequence)
    logo = nucleotide_logo(model, tokenizer, normalized)
    dependency = nucleotide_dependency_map(
        model,
        tokenizer,
        normalized,
        rc=True,
        combine=combine,
        norm_ord=norm_ord,
        batch_size=batch_size,
        atol=atol,
    )
    assert dependency.shape == (len(normalized), len(normalized))
    assert np.isfinite(dependency).all(), "dependency matrix is non-finite"
    assert np.allclose(dependency, dependency.T, atol=1e-6), (
        "dependency matrix is not symmetric"
    )
    return SequenceInterpretation(
        sequence=normalized,
        logo=logo,
        dependency=dependency,
    )
