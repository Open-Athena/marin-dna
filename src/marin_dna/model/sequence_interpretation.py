"""Sequence-direct model interpretation helpers.

The probability-logo computation originated in the branch-only sequence
explorer from issue #387.  This module also exposes the base-alignment steps
needed by the code-visible inference tutorial in issue #409.
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
    """Arrays defining a logo, with matrix columns in A/C/G/T order."""

    probabilities: np.ndarray
    entropy_bits: np.ndarray
    information_bits: np.ndarray
    glyph_heights_bits: np.ndarray


@dataclass(frozen=True)
class AlignedSequenceStrandOutputs:
    """One model strand realigned to forward-sequence base coordinates.

    ``nucleotide_logits`` contains the causal prediction at the token
    immediately before each nucleotide. ``observed_log_probabilities_nats``
    gathers the observed base from a full-vocabulary log-softmax at those same
    causal positions. ``embeddings`` contains the final-layer hidden state at
    the nucleotide token itself.
    """

    nucleotide_logits: np.ndarray
    observed_log_probabilities_nats: np.ndarray
    embeddings: np.ndarray


@dataclass(frozen=True)
class SequenceModelOutputs:
    """Forward/reverse-complement aggregate for one input sequence."""

    logo: NucleotideLogo
    forward_log_likelihood_nats_per_base: float
    reverse_complement_log_likelihood_nats_per_base: float
    average_log_likelihood_nats_per_base: float
    mean_observed_log_probabilities_nats: np.ndarray
    embeddings: np.ndarray


@dataclass(frozen=True)
class SequenceInterpretation:
    """Complete sequence-explorer interpretation."""

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


def align_sequence_strand_outputs(
    input_ids: Tensor,
    logits: Tensor,
    final_hidden_state: Tensor,
    tokenizer: Any,
    sequence_length: int,
    *,
    reverse_complemented: bool,
) -> AlignedSequenceStrandOutputs:
    """Align one strand's full model outputs to forward-sequence bases.

    Inputs may omit or include a singleton batch dimension. The tokenizer must
    prepend at least one special token (the MarinDNA tokenizer prepends BOS), so
    base zero has a causal prediction. When ``reverse_complemented=True``,
    positions are reversed, and only the A/C/G/T logit channels are
    complemented. Hidden states and gathered observed-base log-probabilities
    require only the position reversal.
    """
    assert sequence_length >= 1
    if input_ids.ndim == 2:
        assert input_ids.shape[0] == 1, (
            f"expected singleton input batch, got {tuple(input_ids.shape)}"
        )
        input_ids = input_ids[0]
    if logits.ndim == 3:
        assert logits.shape[0] == 1, (
            f"expected singleton logit batch, got {tuple(logits.shape)}"
        )
        logits = logits[0]
    if final_hidden_state.ndim == 3:
        assert final_hidden_state.shape[0] == 1, (
            "expected singleton hidden-state batch, got "
            f"{tuple(final_hidden_state.shape)}"
        )
        final_hidden_state = final_hidden_state[0]

    assert input_ids.ndim == 1, f"expected input_ids [T], got {tuple(input_ids.shape)}"
    assert logits.ndim == 2, f"expected logits [T,V], got {tuple(logits.shape)}"
    assert final_hidden_state.ndim == 2, (
        f"expected hidden states [T,D], got {tuple(final_hidden_state.shape)}"
    )

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_prefix >= 1, (
        "base-aligned causal predictions require an auto-prepended BOS token"
    )
    expected_tokens = sequence_length + n_prefix + n_suffix
    assert input_ids.shape[0] == expected_tokens, (
        f"encoded {sequence_length} bases as {input_ids.shape[0]} tokens, expected "
        f"{expected_tokens} including {n_prefix} prefix and {n_suffix} suffix tokens"
    )
    assert logits.shape[0] == expected_tokens, (
        f"logit token length {logits.shape[0]} != encoded length {expected_tokens}"
    )
    assert final_hidden_state.shape[0] == expected_tokens, (
        "hidden-state token length "
        f"{final_hidden_state.shape[0]} != encoded length {expected_tokens}"
    )

    device = logits.device
    base_indices = torch.arange(sequence_length, device=device) + n_prefix
    causal_indices = base_indices - 1
    causal_logits = logits[causal_indices].float()
    target_ids = input_ids.to(device=device, dtype=torch.long)[base_indices]
    assert int(target_ids.min()) >= 0 and int(target_ids.max()) < causal_logits.shape[1]

    token_ids_by_nucleotide = _get_nucleotide_token_ids(tokenizer)
    nucleotide_token_ids = torch.tensor(
        [token_ids_by_nucleotide[nuc] for nuc in NUCLEOTIDES],
        dtype=torch.long,
        device=device,
    )
    assert len(set(nucleotide_token_ids.detach().cpu().tolist())) == len(NUCLEOTIDES)
    nucleotide_id_set = set(nucleotide_token_ids.detach().cpu().tolist())
    assert set(target_ids.detach().cpu().tolist()) <= nucleotide_id_set, (
        "encoded DNA positions include a non-nucleotide token"
    )

    nucleotide_logits = causal_logits[:, nucleotide_token_ids]
    observed_log_probabilities = torch.log_softmax(causal_logits, dim=-1).gather(
        1, target_ids[:, None]
    )[:, 0]
    embeddings = final_hidden_state.to(device=device)[base_indices].float()

    nucleotide_logits_np = nucleotide_logits.detach().cpu().numpy()
    observed_log_probabilities_np = observed_log_probabilities.detach().cpu().numpy()
    embeddings_np = embeddings.detach().cpu().numpy()

    if reverse_complemented:
        nucleotide_logits_np = nucleotide_logits_np[::-1, _RC_CHANNEL_INDICES].copy()
        observed_log_probabilities_np = observed_log_probabilities_np[::-1].copy()
        embeddings_np = embeddings_np[::-1].copy()

    assert nucleotide_logits_np.shape == (sequence_length, len(NUCLEOTIDES))
    assert observed_log_probabilities_np.shape == (sequence_length,)
    assert embeddings_np.shape[0] == sequence_length and embeddings_np.ndim == 2
    assert np.isfinite(nucleotide_logits_np).all(), "non-finite nucleotide logits"
    assert np.isfinite(observed_log_probabilities_np).all(), (
        "non-finite observed-base log-probabilities"
    )
    assert np.isfinite(embeddings_np).all(), "non-finite per-position embeddings"
    return AlignedSequenceStrandOutputs(
        nucleotide_logits=nucleotide_logits_np,
        observed_log_probabilities_nats=observed_log_probabilities_np,
        embeddings=embeddings_np,
    )


@torch.inference_mode()
def run_aligned_sequence_strand(
    model: Any,
    tokenizer: Any,
    input_ids: Tensor,
    sequence_length: int,
    *,
    reverse_complemented: bool,
) -> AlignedSequenceStrandOutputs:
    """Run one model strand and align its readouts to forward coordinates."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1, (
        f"expected singleton input batch [1,T], got {tuple(input_ids.shape)}"
    )

    model_output = model(
        input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    assert model_output.logits.shape[:2] == input_ids.shape
    assert model_output.hidden_states is not None
    final_hidden_state = model_output.hidden_states[-1]
    assert final_hidden_state.shape[:2] == input_ids.shape
    return align_sequence_strand_outputs(
        input_ids,
        model_output.logits,
        final_hidden_state,
        tokenizer,
        sequence_length,
        reverse_complemented=reverse_complemented,
    )


def _nucleotide_logo_from_probabilities(probabilities: np.ndarray) -> NucleotideLogo:
    assert probabilities.ndim == 2 and probabilities.shape[1] == len(NUCLEOTIDES)
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


def nucleotide_logo_from_aligned_logits(
    forward_logits: np.ndarray,
    reverse_complement_logits: np.ndarray,
) -> NucleotideLogo:
    """Average aligned A/C/G/T logits, then apply the four-base softmax."""
    forward_logits = np.asarray(forward_logits, dtype=np.float32)
    reverse_complement_logits = np.asarray(reverse_complement_logits, dtype=np.float32)
    assert forward_logits.shape == reverse_complement_logits.shape
    assert forward_logits.ndim == 2
    assert forward_logits.shape[1] == len(NUCLEOTIDES)
    assert np.isfinite(forward_logits).all()
    assert np.isfinite(reverse_complement_logits).all()
    mean_logits = (forward_logits + reverse_complement_logits) / 2.0
    probabilities = torch.softmax(torch.from_numpy(mean_logits), dim=-1).numpy()
    return _nucleotide_logo_from_probabilities(probabilities)


def aggregate_sequence_strands(
    forward: AlignedSequenceStrandOutputs,
    reverse_complement: AlignedSequenceStrandOutputs,
) -> SequenceModelOutputs:
    """Average two already aligned strands without mixing readout conventions."""
    assert forward.nucleotide_logits.shape == reverse_complement.nucleotide_logits.shape
    assert (
        forward.observed_log_probabilities_nats.shape
        == reverse_complement.observed_log_probabilities_nats.shape
    )
    assert forward.embeddings.shape == reverse_complement.embeddings.shape

    logo = nucleotide_logo_from_aligned_logits(
        forward.nucleotide_logits,
        reverse_complement.nucleotide_logits,
    )
    forward_log_likelihood = float(
        np.mean(forward.observed_log_probabilities_nats, dtype=np.float64)
    )
    reverse_log_likelihood = float(
        np.mean(
            reverse_complement.observed_log_probabilities_nats,
            dtype=np.float64,
        )
    )
    average_log_likelihood = (forward_log_likelihood + reverse_log_likelihood) / 2.0
    mean_observed_log_probabilities = (
        np.asarray(forward.observed_log_probabilities_nats, dtype=np.float32)
        + np.asarray(
            reverse_complement.observed_log_probabilities_nats,
            dtype=np.float32,
        )
    ) / 2.0
    embeddings = (
        np.asarray(forward.embeddings, dtype=np.float32)
        + np.asarray(reverse_complement.embeddings, dtype=np.float32)
    ) / 2.0
    assert np.isfinite(mean_observed_log_probabilities).all()
    assert np.isfinite(embeddings).all()
    return SequenceModelOutputs(
        logo=logo,
        forward_log_likelihood_nats_per_base=forward_log_likelihood,
        reverse_complement_log_likelihood_nats_per_base=reverse_log_likelihood,
        average_log_likelihood_nats_per_base=average_log_likelihood,
        mean_observed_log_probabilities_nats=mean_observed_log_probabilities,
        embeddings=embeddings,
    )


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
        "sequence interpretation requires one token per nucleotide: encoded "
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
    return nucleotide_logo_from_aligned_logits(
        forward_logits.cpu().numpy(),
        reverse_in_forward_coordinates.cpu().numpy(),
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
    """Compute both sequence-explorer views from one validated DNA sequence."""
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
