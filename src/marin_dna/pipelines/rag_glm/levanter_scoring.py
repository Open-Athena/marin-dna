"""Levanter paged-KV scorer for issue #402's online RAG evaluation.

The generic lm-eval loglikelihood path forwards reference and alternate strings
independently.  This focused scorer instead prefills the fixed RAG prefix once,
shares its page-aligned KV pages between two sequence slots, and force-scores
the paired completions.  It deliberately does not modify Levanter's generic
evaluation kernel or MarinDNA's existing short-window scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marin_dna.pipelines.rag_glm.hf_scoring import (
    RAG_COMPLETION_TOKENS,
    RAG_PREFIX_TOKENS,
)


@dataclass(frozen=True)
class RagCacheExecutionPlan:
    """Static token/slot schedule used by the online cached scorer."""

    prefix_positions: tuple[int, ...]
    prefix_slots: tuple[int, ...]
    continuation_tokens_per_branch: int
    continuation_positions: tuple[int, ...]
    continuation_slots: tuple[int, ...]
    page_size: int


def build_rag_cache_execution_plan(
    *,
    prefix_length: int = RAG_PREFIX_TOKENS,
    completion_length: int = RAG_COMPLETION_TOKENS,
    page_size: int = 128,
) -> RagCacheExecutionPlan:
    """Build the one-prefix/two-continuation paged-cache schedule."""
    assert prefix_length > 0
    assert completion_length > 1
    assert page_size > 0
    assert prefix_length % page_size == 0, (
        "the shared prefix must end at a KV page boundary so both branches can "
        "share every prefix page without copying a partially filled page"
    )
    continuation_positions = tuple(
        range(prefix_length, prefix_length + completion_length - 1)
    )
    return RagCacheExecutionPlan(
        prefix_positions=tuple(range(prefix_length)),
        prefix_slots=(0,) * prefix_length,
        continuation_tokens_per_branch=completion_length - 1,
        continuation_positions=continuation_positions * 2,
        continuation_slots=(0,) * (completion_length - 1)
        + (1,) * (completion_length - 1),
        page_size=page_size,
    )


def _jax_nucleotide_indices(token_ids: Any, nucleotide_token_ids: Any) -> Any:
    import equinox as eqx
    import jax.numpy as jnp

    matches = token_ids[..., None] == nucleotide_token_ids
    token_ids = eqx.error_if(
        token_ids,
        ~jnp.all(jnp.sum(matches, axis=-1) == 1),
        "all RAG completion targets must map to exactly one A/C/G/T token",
    )
    matches = token_ids[..., None] == nucleotide_token_ids
    return jnp.argmax(matches, axis=-1)


def score_rag_completions_levanter(
    model: Any,
    prefix_ids: Any,
    ref_completion_ids: Any,
    alt_completion_ids: Any,
    nucleotide_token_ids: Any,
    *,
    prefix_length: int = RAG_PREFIX_TOKENS,
    completion_length: int = RAG_COMPLETION_TOKENS,
    page_size: int = 128,
    compute_dtype: Any = None,
) -> Any:
    """Return ``[ref_loglikelihood, alt_loglikelihood, raw_llr]``.

    Inputs are one-dimensional JAX arrays.  The function is intended to be
    wrapped once with ``haliax.named_jit`` by the lm-eval adapter.  Scores use
    the four-nucleotide softmax, include the variant and all 127 downstream
    targets, and have no length normalization.
    """
    import haliax as hax
    import jax
    import jax.numpy as jnp
    from levanter.inference.jit_scheduler import DecodeState
    from levanter.inference.page_table import PageTable

    if compute_dtype is None:
        compute_dtype = jnp.bfloat16

    plan = build_rag_cache_execution_plan(
        prefix_length=prefix_length,
        completion_length=completion_length,
        page_size=page_size,
    )
    assert prefix_ids.shape == (prefix_length,)
    assert ref_completion_ids.shape == (completion_length,)
    assert alt_completion_ids.shape == (completion_length,)
    assert nucleotide_token_ids.shape == (4,)
    assert prefix_length + completion_length <= model.max_length

    same_downstream = jnp.all(ref_completion_ids[1:] == alt_completion_ids[1:])
    different_alleles = ref_completion_ids[0] != alt_completion_ids[0]
    import equinox as eqx

    ref_completion_ids = eqx.error_if(
        ref_completion_ids,
        ~same_downstream,
        "reference and alternate RAG completions may differ only at the SNV",
    )
    ref_completion_ids = eqx.error_if(
        ref_completion_ids,
        ~different_alleles,
        "reference and alternate RAG alleles must differ",
    )
    _jax_nucleotide_indices(ref_completion_ids, nucleotide_token_ids)
    _jax_nucleotide_indices(alt_completion_ids, nucleotide_token_ids)

    max_pages_per_sequence = (model.max_length + page_size - 1) // page_size
    # Worst case allows both branches to own every page; the page-aligned
    # prefix actually shares its pages, leaving ample defensive headroom.
    page_table = PageTable.init(
        max_pages=2 * max_pages_per_sequence,
        max_seqs=2,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_sequence,
    )
    decode_state = DecodeState.init(page_table=page_table, max_queued_tokens=0)
    decode_state, _ = decode_state.reserve_slot(0)
    cache = model.initial_cache(page_table.spec(), dtype=compute_dtype)

    prefix_tokens = hax.named(prefix_ids, "position")
    prefix_slots = hax.named(
        jnp.asarray(plan.prefix_slots, dtype=jnp.int32), "position"
    )
    prefix_positions = hax.named(
        jnp.asarray(plan.prefix_positions, dtype=jnp.int32), "position"
    )
    decode_state, prefix_batch = decode_state.allocate_for_seq(
        token_slot_ids=prefix_slots,
        token_pos_ids=prefix_positions,
    )
    prefix_logits, cache = model.decode(
        prefix_tokens,
        cache,
        prefix_batch,
        prefix_positions,
    )

    # 1,920 is exactly 15 × 128.  All prefix pages can therefore be shared;
    # allocation of position 1,920 gives each branch a new continuation page.
    decode_state = decode_state.clone_pages_from(0, 1)
    continuation_tokens = jnp.concatenate(
        [ref_completion_ids[:-1], alt_completion_ids[:-1]], axis=0
    )
    continuation_tokens = hax.named(continuation_tokens, "position")
    continuation_slots = hax.named(
        jnp.asarray(plan.continuation_slots, dtype=jnp.int32), "position"
    )
    continuation_positions = hax.named(
        jnp.asarray(plan.continuation_positions, dtype=jnp.int32), "position"
    )
    decode_state, continuation_batch = decode_state.allocate_for_seq(
        token_slot_ids=continuation_slots,
        token_pos_ids=continuation_positions,
    )
    continuation_logits, _ = model.decode(
        continuation_tokens,
        cache,
        continuation_batch,
        continuation_positions,
    )

    prefix_array = prefix_logits.rearrange(("position", model.Vocab)).array
    prefix_nucleotide_logits = prefix_array[-1, nucleotide_token_ids]
    prefix_log_probs = jax.nn.log_softmax(
        prefix_nucleotide_logits.astype(jnp.float32), axis=-1
    )
    first_targets = jnp.stack([ref_completion_ids[0], alt_completion_ids[0]], axis=0)
    first_target_indices = _jax_nucleotide_indices(first_targets, nucleotide_token_ids)
    first_log_probs = prefix_log_probs[first_target_indices]

    continuation_array = continuation_logits.rearrange(("position", model.Vocab)).array
    branch_logits = continuation_array.reshape(
        2, completion_length - 1, model.Vocab.size
    )
    nucleotide_logits = branch_logits[..., nucleotide_token_ids]
    continuation_log_probs = jax.nn.log_softmax(
        nucleotide_logits.astype(jnp.float32), axis=-1
    )
    downstream_targets = jnp.stack(
        [ref_completion_ids[1:], alt_completion_ids[1:]], axis=0
    )
    downstream_target_indices = _jax_nucleotide_indices(
        downstream_targets, nucleotide_token_ids
    )
    downstream_log_probs = jnp.take_along_axis(
        continuation_log_probs,
        downstream_target_indices[..., None],
        axis=-1,
    )[..., 0]
    paired_loglikelihoods = first_log_probs + jnp.sum(downstream_log_probs, axis=-1)
    return jnp.stack(
        [
            paired_loglikelihoods[0],
            paired_loglikelihoods[1],
            paired_loglikelihoods[1] - paired_loglikelihoods[0],
        ]
    )


def score_rag_batch_levanter(
    model: Any,
    prefix_ids: Any,
    ref_completion_ids: Any,
    alt_completion_ids: Any,
    nucleotide_token_ids: Any,
    *,
    prefix_length: int = RAG_PREFIX_TOKENS,
    completion_length: int = RAG_COMPLETION_TOKENS,
    page_size: int = 128,
    compute_dtype: Any = None,
) -> Any:
    """Vectorize the paired paged-cache scorer over a fixed host batch."""
    import jax

    assert prefix_ids.ndim == 2
    assert ref_completion_ids.ndim == 2
    assert alt_completion_ids.ndim == 2
    assert prefix_ids.shape[0] == ref_completion_ids.shape[0]
    assert prefix_ids.shape[0] == alt_completion_ids.shape[0]
    return jax.vmap(
        lambda prefix, ref, alt: score_rag_completions_levanter(
            model,
            prefix,
            ref,
            alt,
            nucleotide_token_ids,
            prefix_length=prefix_length,
            completion_length=completion_length,
            page_size=page_size,
            compute_dtype=compute_dtype,
        )
    )(prefix_ids, ref_completion_ids, alt_completion_ids)


def score_rag_completions_naive_levanter(
    model: Any,
    prefix_ids: Any,
    ref_completion_ids: Any,
    alt_completion_ids: Any,
    nucleotide_token_ids: Any,
    *,
    prefix_length: int = RAG_PREFIX_TOKENS,
    completion_length: int = RAG_COMPLETION_TOKENS,
) -> Any:
    """Reference-score two complete documents without a KV cache."""
    import haliax as hax
    import jax
    import jax.numpy as jnp
    from levanter.layers.attention import AttentionMask

    document_length = prefix_length + completion_length
    assert prefix_ids.shape == (prefix_length,)
    assert ref_completion_ids.shape == (completion_length,)
    assert alt_completion_ids.shape == (completion_length,)
    assert nucleotide_token_ids.shape == (4,)
    assert document_length <= model.max_length

    def score_completion(completion_ids: Any) -> Any:
        tokens = jnp.concatenate([prefix_ids, completion_ids])
        position = hax.Axis("position", document_length)
        logits = (
            model(
                hax.named(tokens, position),
                attn_mask=AttentionMask.causal(),
                pos_ids=hax.named(
                    jnp.arange(document_length, dtype=jnp.int32), position
                ),
                key=None,
            )
            .rearrange((position, model.Vocab))
            .array
        )
        nucleotide_logits = logits[prefix_length - 1 : -1, nucleotide_token_ids]
        log_probs = jax.nn.log_softmax(nucleotide_logits.astype(jnp.float32), axis=-1)
        target_indices = _jax_nucleotide_indices(completion_ids, nucleotide_token_ids)
        return jnp.take_along_axis(log_probs, target_indices[:, None], axis=-1).sum()

    ref_loglikelihood = score_completion(ref_completion_ids)
    alt_loglikelihood = score_completion(alt_completion_ids)
    return jnp.stack(
        [
            ref_loglikelihood,
            alt_loglikelihood,
            alt_loglikelihood - ref_loglikelihood,
        ]
    )


def score_rag_batch_naive_levanter(
    model: Any,
    prefix_ids: Any,
    ref_completion_ids: Any,
    alt_completion_ids: Any,
    nucleotide_token_ids: Any,
) -> Any:
    """Vectorize the full-forward reference scorer over a small debug batch."""
    import jax

    assert prefix_ids.ndim == 2
    assert ref_completion_ids.ndim == 2
    assert alt_completion_ids.ndim == 2
    assert prefix_ids.shape[0] == ref_completion_ids.shape[0]
    assert prefix_ids.shape[0] == alt_completion_ids.shape[0]
    return jax.vmap(
        lambda prefix, ref, alt: score_rag_completions_naive_levanter(
            model,
            prefix,
            ref,
            alt,
            nucleotide_token_ids,
        )
    )(prefix_ids, ref_completion_ids, alt_completion_ids)
