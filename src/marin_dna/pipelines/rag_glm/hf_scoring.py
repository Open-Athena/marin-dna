"""Hugging Face scorer for issue #402's materialized RAG VEP rows.

This is intentionally separate from :mod:`marin_dna.model.scoring`.  The RAG
prototype has a fixed 1,920-token shared prefix and two explicit 128-token
completions; keeping the focused cache logic here makes that contract visible
and leaves the existing short-window scoring pipeline unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from transformers.cache_utils import DynamicCache

RAG_PREFIX_TOKENS = 1_920
RAG_COMPLETION_TOKENS = 128
RAG_DOCUMENT_TOKENS = RAG_PREFIX_TOKENS + RAG_COMPLETION_TOKENS
RAG_VARIANT_TOKEN_INDEX = RAG_PREFIX_TOKENS
RAG_HUMAN_POOL_START = 1_793
RAG_HUMAN_POOL_TOKENS = 255


def _token_id_to_nucleotide_index(
    token_ids: Int[Tensor, "..."],
    nucleotide_token_ids: Int[Tensor, " 4"],
) -> Int[Tensor, "..."]:
    matches = token_ids.unsqueeze(-1) == nucleotide_token_ids
    assert bool(matches.any(dim=-1).all()), "all scored targets must be A/C/G/T tokens"
    assert bool((matches.sum(dim=-1) == 1).all()), "nucleotide token IDs must be unique"
    return matches.to(torch.int64).argmax(dim=-1)


def _duplicate_rag_kv_cache(past_key_values: Any) -> Any:
    """Duplicate one prefix cache into adjacent reference/alternate branches."""
    if hasattr(past_key_values, "batch_repeat_interleave"):
        past_key_values.batch_repeat_interleave(2)
        return past_key_values

    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        for index in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[index] = past_key_values.key_cache[
                index
            ].repeat_interleave(2, dim=0)
            past_key_values.value_cache[index] = past_key_values.value_cache[
                index
            ].repeat_interleave(2, dim=0)
        return past_key_values

    duplicated = DynamicCache()
    for layer_index, (keys, values) in enumerate(past_key_values):
        duplicated.update(
            keys.repeat_interleave(2, dim=0),
            values.repeat_interleave(2, dim=0),
            layer_index,
        )
    return duplicated


def score_rag_completions_hf(
    model: Any,
    prefix_ids: Int[Tensor, "B P"],
    ref_completion_ids: Int[Tensor, "B C"],
    alt_completion_ids: Int[Tensor, "B C"],
    *,
    nucleotide_token_ids: Int[Tensor, " 4"],
    return_embeddings: bool = False,
    expected_prefix_tokens: int = RAG_PREFIX_TOKENS,
) -> Float[Tensor, "B W"]:
    """Score paired RAG completions with one prefix forward per document.

    The returned columns are ``ref_loglikelihood``, ``alt_loglikelihood``, and
    raw ``llr = alt - ref``.  Every probability is normalized over the four
    nucleotide tokens, matching the established MarinDNA VEP protocol.  No
    length normalization is applied.

    The batch is ordered ``(doc0/ref, doc0/alt, doc1/ref, doc1/alt, ...)`` so
    ``repeat_interleave(2)`` produces a byte-for-byte corresponding prefix KV
    cache for each continuation branch.
    """
    assert prefix_ids.ndim == 2
    assert ref_completion_ids.ndim == 2
    assert alt_completion_ids.ndim == 2
    batch_size, prefix_length = prefix_ids.shape
    assert expected_prefix_tokens > 0
    assert prefix_length == expected_prefix_tokens
    assert ref_completion_ids.shape == (batch_size, RAG_COMPLETION_TOKENS)
    assert alt_completion_ids.shape == ref_completion_ids.shape
    assert nucleotide_token_ids.shape == (4,)
    assert nucleotide_token_ids.unique().numel() == 4
    assert bool((ref_completion_ids[:, 1:] == alt_completion_ids[:, 1:]).all()), (
        "reference and alternate completions may differ only at the SNV"
    )
    assert bool((ref_completion_ids[:, 0] != alt_completion_ids[:, 0]).all()), (
        "reference and alternate alleles must differ"
    )
    assert prefix_length + ref_completion_ids.shape[1] <= RAG_DOCUMENT_TOKENS
    if return_embeddings:
        assert expected_prefix_tokens == RAG_PREFIX_TOKENS, (
            "human-segment embeddings require the fixed 2,048-token geometry"
        )

    device_nucleotides = nucleotide_token_ids.to(prefix_ids.device)
    _token_id_to_nucleotide_index(ref_completion_ids, device_nucleotides)
    _token_id_to_nucleotide_index(alt_completion_ids, device_nucleotides)

    # Capture only the final decoder layer when embeddings are requested. A
    # base-model hook avoids materializing every layer via output_hidden_states.
    hidden_capture: list[Tensor] = []
    hook = (
        model.base_model.register_forward_hook(
            lambda _module, _inputs, output: hidden_capture.append(
                output.last_hidden_state
            )
        )
        if return_embeddings
        else None
    )
    try:
        # Exactly one model call computes every document's complete shared prefix.
        prefix_output = model(prefix_ids, use_cache=True, logits_to_keep=1)
        prefix_logits = prefix_output.logits[:, -1]
        paired_cache = _duplicate_rag_kv_cache(prefix_output.past_key_values)

        completions = torch.stack(
            [ref_completion_ids, alt_completion_ids], dim=1
        ).flatten(0, 1)
        continuation_output = model(
            completions,
            past_key_values=paired_cache,
            use_cache=False,
        )
    finally:
        if hook is not None:
            hook.remove()
    continuation_logits = continuation_output.logits.unflatten(0, (batch_size, 2))

    prefix_log_probs = F.log_softmax(
        prefix_logits[..., device_nucleotides].float(), dim=-1
    )
    first_targets = torch.stack(
        [ref_completion_ids[:, 0], alt_completion_ids[:, 0]], dim=1
    )
    first_target_indices = _token_id_to_nucleotide_index(
        first_targets, device_nucleotides
    )
    first_log_probs = (
        prefix_log_probs.unsqueeze(1)
        .expand(-1, 2, -1)
        .gather(-1, first_target_indices.unsqueeze(-1))
        .squeeze(-1)
    )

    # Logit position i predicts completion token i+1.  The final logit predicts
    # outside the frozen document and is deliberately excluded.
    continuation_log_probs = F.log_softmax(
        continuation_logits[:, :, :-1, device_nucleotides].float(), dim=-1
    )
    downstream_targets = torch.stack(
        [ref_completion_ids[:, 1:], alt_completion_ids[:, 1:]], dim=1
    )
    downstream_target_indices = _token_id_to_nucleotide_index(
        downstream_targets, device_nucleotides
    )
    downstream_log_probs = continuation_log_probs.gather(
        -1, downstream_target_indices.unsqueeze(-1)
    ).squeeze(-1)

    paired_loglikelihoods = first_log_probs + downstream_log_probs.sum(dim=-1)
    ref_loglikelihood = paired_loglikelihoods[:, 0]
    alt_loglikelihood = paired_loglikelihoods[:, 1]
    scores = torch.stack(
        [
            ref_loglikelihood,
            alt_loglikelihood,
            alt_loglikelihood - ref_loglikelihood,
        ],
        dim=-1,
    )
    if not return_embeddings:
        return scores

    assert RAG_PREFIX_TOKENS - RAG_HUMAN_POOL_START == 127
    assert RAG_DOCUMENT_TOKENS - RAG_HUMAN_POOL_START == RAG_HUMAN_POOL_TOKENS
    assert len(hidden_capture) == 2, (
        f"expected prefix and continuation hidden states, got {len(hidden_capture)}"
    )
    prefix_hidden = hidden_capture[0]
    continuation_hidden = hidden_capture[1].unflatten(0, (batch_size, 2))
    assert prefix_hidden.shape[:2] == (batch_size, RAG_PREFIX_TOKENS)
    assert continuation_hidden.shape[:3] == (
        batch_size,
        2,
        RAG_COMPLETION_TOKENS,
    )
    prefix_human_sum = prefix_hidden[:, RAG_HUMAN_POOL_START:].sum(
        dim=1, dtype=torch.float32
    )
    ref_human_sum = prefix_human_sum + continuation_hidden[:, 0].sum(
        dim=1, dtype=torch.float32
    )
    alt_human_sum = prefix_human_sum + continuation_hidden[:, 1].sum(
        dim=1, dtype=torch.float32
    )
    emb_ref = ref_human_sum / RAG_HUMAN_POOL_TOKENS
    emb_alt = alt_human_sum / RAG_HUMAN_POOL_TOKENS
    return torch.cat([scores, emb_ref, emb_alt], dim=-1)


def score_rag_completions_naive_hf(
    model: Any,
    prefix_ids: Int[Tensor, "B P"],
    ref_completion_ids: Int[Tensor, "B C"],
    alt_completion_ids: Int[Tensor, "B C"],
    *,
    nucleotide_token_ids: Int[Tensor, " 4"],
    expected_prefix_tokens: int = RAG_PREFIX_TOKENS,
) -> Float[Tensor, "B 3"]:
    """Reference-score complete paired documents without a KV cache."""
    assert prefix_ids.ndim == 2
    batch_size, prefix_length = prefix_ids.shape
    assert expected_prefix_tokens > 0
    assert prefix_length == expected_prefix_tokens
    assert ref_completion_ids.shape == (batch_size, RAG_COMPLETION_TOKENS)
    assert alt_completion_ids.shape == ref_completion_ids.shape
    assert bool((ref_completion_ids[:, 1:] == alt_completion_ids[:, 1:]).all())
    assert bool((ref_completion_ids[:, 0] != alt_completion_ids[:, 0]).all())

    device_nucleotides = nucleotide_token_ids.to(prefix_ids.device)
    completions = torch.stack([ref_completion_ids, alt_completion_ids], dim=1).flatten(
        0, 1
    )
    documents = torch.cat([prefix_ids.repeat_interleave(2, dim=0), completions], dim=1)
    document_tokens = expected_prefix_tokens + RAG_COMPLETION_TOKENS
    assert document_tokens <= RAG_DOCUMENT_TOKENS
    assert documents.shape == (batch_size * 2, document_tokens)
    logits = model(documents, use_cache=False).logits
    nucleotide_logits = logits[:, expected_prefix_tokens - 1 : -1, device_nucleotides]
    log_probs = F.log_softmax(nucleotide_logits.float(), dim=-1)
    target_indices = _token_id_to_nucleotide_index(completions, device_nucleotides)
    loglikelihoods = (
        log_probs.gather(-1, target_indices.unsqueeze(-1))
        .squeeze(-1)
        .sum(dim=-1)
        .unflatten(0, (batch_size, 2))
    )
    ref_loglikelihood = loglikelihoods[:, 0]
    alt_loglikelihood = loglikelihoods[:, 1]
    return torch.stack(
        [
            ref_loglikelihood,
            alt_loglikelihood,
            alt_loglikelihood - ref_loglikelihood,
        ],
        dim=-1,
    )
