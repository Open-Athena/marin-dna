"""CLM scoring kernels.

Vendored from biofoundation/model/scoring.py at commit 834dd4c (May 2026),
CLM-only (MLM compute functions dropped). Rewritten to call HF model
methods directly (``model(input_ids).logits``, ``model(input_ids,
output_hidden_states=True).hidden_states[i]``) — no ``CausalLM`` /
``CausalLMWithEmbeddings`` / ``EmbeddingModel`` abstract base classes.

The ``model`` argument is duck-typed: any callable whose output exposes
``.logits`` (and ``.hidden_states`` when ``output_hidden_states=True`` is
passed) works. HF ``AutoModelForCausalLM`` satisfies this natively;
non-HF models (e.g. Evo2) should be wrapped to expose the same surface —
see ``pipelines/evals/evo2.py`` for an example.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.cache_utils import DynamicCache


# https://github.com/ArcInstitute/evo2/blob/4c3c8522dc99d2dc14b5b5a07cd65f2b67e6f457/evo2/scoring.py#L37
def _logits_to_logprobs(
    logits: Float[Tensor, "B L V"],
    input_ids: Int[Tensor, "B L"],
) -> Float[Tensor, "B L-1"]:
    """Per-token log-likelihoods of the provided sequence at each position.

    Takes logits ``[B, L, V]`` and uses ``input_ids`` to index into the log-
    likelihoods, returning ``[B, L-1]``.

    Logits are cast to fp32 before log_softmax: bf16 log_softmax has
    ~10^-3 per-token rounding error that compounds across the sequence
    sum in ``_clm_seq_logprob`` (see biofoundation issue #21).
    """
    softmax_logprobs = torch.log_softmax(logits.float(), dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    input_ids = input_ids[:, 1:]
    assert softmax_logprobs.shape[1] == input_ids.shape[1]

    logprobs = torch.gather(
        softmax_logprobs,  # Gather likelihoods...
        2,  # along the vocab dimension...
        input_ids.unsqueeze(-1),  # using the token ids to index.
    ).squeeze(-1)

    return logprobs


def _clm_seq_logprob(
    logits: Float[Tensor, "B L V"],
    input_ids: Int[Tensor, "B L"],
) -> Float[Tensor, " B"]:
    log_probs = _logits_to_logprobs(logits, input_ids)
    return reduce(log_probs.float(), "B L -> B", "sum")


def compute_reflogprob_clm(
    model: Any,
    input_ids: Int[Tensor, "B 4 L"],
    ref: Int[Tensor, " B"],
) -> Float[Tensor, " B"]:
    B = input_ids.shape[0]
    batch_indices = torch.arange(B)
    input_ids = rearrange(input_ids, "B V L -> (B V) L")
    logits = model(input_ids).logits
    log_prob = _clm_seq_logprob(logits, input_ids)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)
    # marginal log-probability of each of the 4 alleles
    marginal_log_prob = torch.log_softmax(log_prob, dim=-1)
    ref_log_prob = marginal_log_prob[batch_indices, ref]
    return ref_log_prob


def compute_ll_clm(
    model: Any,
    input_ids: Int[Tensor, "B L"],
    is_upper: Bool[Tensor, "B L"] | None = None,
) -> Float[Tensor, "B 2"] | Float[Tensor, "B 4"]:
    """Per-sequence log-likelihood sums and target counts under a CLM.

    Returns sums and counts (not means) so callers can aggregate to a
    dataset-wide token-weighted mean LL by summing across rows then
    dividing — correct even for sequences that are all-upper or
    all-lower in their case mask.

    ``_logits_to_logprobs`` returns ``[B, L-1]`` where entry ``[b, i]``
    is ``log p(input_ids[b, i+1] | input_ids[b, :i+1])``, so when an
    ``is_upper`` mask is supplied, the relevant case is the case of the
    *target* ``input_ids[i+1]`` — we slice ``is_upper[:, 1:]`` to align
    with the L-1 log-probs.

    Output:

    - Without ``is_upper``: ``[B, 2]`` of ``(ll_sum, n)`` per row.
      ``n = L - 1`` for every row.
    - With ``is_upper``: ``[B, 4]`` of
      ``(ll_sum_upper, ll_sum_lower, n_upper, n_lower)`` per row.
      Invariants: ``ll_sum_upper + ll_sum_lower`` is the total sum,
      ``n_upper + n_lower = L - 1``. Special-token target positions
      (``is_upper = False``) fall into the "lower" bucket.

    Per-row sums are fp32; aggregating across many rows can exceed fp32
    precision (~0.5 absolute error at totals of ~10^6, reachable on a
    16k-row eval set), so cast to fp64 before the cross-row sum.
    """
    logits = model(input_ids).logits
    logp = _logits_to_logprobs(logits, input_ids).float()  # [B, L-1]
    L_minus_1 = logp.shape[-1]
    ll_sum_total = logp.sum(dim=-1)
    if is_upper is None:
        n = torch.full_like(ll_sum_total, float(L_minus_1))
        return torch.stack([ll_sum_total, n], dim=-1)
    upper_t = is_upper[:, 1:].float()
    n_upper = upper_t.sum(dim=-1)
    ll_sum_upper = (logp * upper_t).sum(dim=-1)
    ll_sum_lower = ll_sum_total - ll_sum_upper
    n_lower = float(L_minus_1) - n_upper
    return torch.stack([ll_sum_upper, ll_sum_lower, n_upper, n_lower], dim=-1)


def compute_euclidean_distance(
    model: Any,
    input_ids: Int[Tensor, "B 2 L"],
) -> Float[Tensor, " B"]:
    """Compute Euclidean distance between reference and alternate embeddings.

    ``model`` must be a callable that, given ``input_ids`` ``[B*2, L]``,
    returns an embeddings tensor of shape ``[B*2, L, D]``. For HF, this
    means passing the base ``AutoModel`` (not the causal-LM head).

    Returns euclidean distance of shape ``[B]``.
    """
    B = input_ids.shape[0]
    input_ids = rearrange(input_ids, "B V L -> (B V) L")
    embeddings = model(input_ids)
    embeddings = rearrange(embeddings, "(B V) L D -> B V (L D)", B=B)
    ref_emb = embeddings[:, 0, :]
    alt_emb = embeddings[:, 1, :]
    return F.pairwise_distance(ref_emb, alt_emb)


def compute_variant_score_bundle(
    model: Any,
    input_ids: Int[Tensor, "B L"],
    alt_token_id: Int[Tensor, " B"],
    *,
    var_pos: int,
    nuc_token_ids: Int[Tensor, " 4"],
) -> Float[Tensor, "B 2"]:
    """Compute LLR and per-position next-token JSD using prefix-sharing.

    SNV-only kernel. For each row, the alt sequence equals ``input_ids`` with
    a single token replaced at ``var_pos`` by ``alt_token_id``. The shared
    prefix ``input_ids[:, :var_pos]`` is forwarded once with KV-cache; the
    two divergent suffixes (ref and alt, length ``L - var_pos``) are then
    forwarded with the cached prefix as context. Same trick as
    lm-eval-harness and inference servers like vLLM.

    Both LLR and JSD operate in the **4-nucleotide softmax** space (rather
    than full vocab). For SNVs both ref and alt targets are always
    nucleotides, so ``log P_full(token | context) = log P_4nuc(token | context,
    nuc) + log P_full(nuc | context)``; the second term cancels exactly at
    the variant position (shared context) and is ~0 elsewhere for any
    well-trained DNA model. Empirically validated against the prior
    full-vocab kernel on exp166-p1B × mendelian: per-row LLR diff is at
    bf16-noise scale (~1e-3) and Global PA shifts by < 0.002. The 4-nuc
    softmax is shared between LLR (gather at the actual nuc index) and JSD
    (full distribution + symmetric KL).

    The kernel skips the LLR computation at prefix positions before
    ``var_pos`` — they contribute zero to the alt-vs-ref log-prob diff
    (same context, same target) — and at ``var_pos - 1`` it gathers from a
    single shared prefix logit. Combined with prefix-sharing this means we
    never materialize ``[B*2, L, V]`` logits.

    ``var_pos`` is a Python int — constant per inference call, derived in
    the wrapper from ``window_size``, ``strand``, and tokenizer
    ``n_prefix`` (BOS). Passing it as a kwarg avoids a ``.item()`` call
    that would graph-break under torch.compile.

    Args:
        model: HF-shaped causal LM. ``model(input_ids, use_cache=True,
            logits_to_keep=N)`` must return ``.logits`` (shape ``[B, N, V]``
            when ``logits_to_keep=N``, else ``[B, L, V]``) and
            ``.past_key_values`` (HF ``Cache`` or legacy tuple).
            ``model(input_ids, past_key_values=...)`` must accept the cache.
        input_ids: Reference sequences, shape ``[B, L]``.
        alt_token_id: Alt nucleotide token ID per row, shape ``[B]``.
        var_pos: Token-level variant position (Python int, constant within batch).
        nuc_token_ids: Length-4 tensor of token IDs for A/C/G/T in
            ``NUCLEOTIDES`` order.

    Returns:
        Tensor with shape ``[B, 2]``:
            - [:, 0]: LLR (alt_logprob - ref_logprob, 4-nuc-softmax space)
            - [:, 1]: next_token_jsd_mean (mean per-position 4-nuc JSD over downstream positions)
    """
    B, L = input_ids.shape
    p = var_pos
    assert 0 < p < L - 1, (
        f"variant at token position {p} of length-{L} sequence has no shared "
        f"prefix or no downstream prediction; expected 0 < var_pos < L-1"
    )

    # Split: shared prefix, divergent suffixes (alt = ref with one token swap at p).
    # Build alt_suffix functionally with torch.cat instead of clone+in-place
    # assignment — avoids the clone allocation and stays compile-friendly
    # (in-place mutation on a freshly-cloned tensor can graph-break on older
    # torch.compile).
    prefix = input_ids[:, :p].contiguous()
    ref_suffix = input_ids[:, p:].contiguous()
    alt_suffix = torch.cat([alt_token_id.unsqueeze(-1), ref_suffix[:, 1:]], dim=-1)
    suffixes = torch.stack([ref_suffix, alt_suffix], dim=1)  # [B, 2, L-p]
    suffixes_flat = rearrange(suffixes, "B V L -> (B V) L").contiguous()

    # 1. Prefix forward — only need logits at the last prefix position
    #    (predicts the variant token); skip the lm_head for the rest.
    prefix_out = model(prefix, use_cache=True, logits_to_keep=1)
    prefix_last_logits = prefix_out.logits[:, -1]  # [B, V]
    past_kv = _repeat_interleave_kv_cache(prefix_out.past_key_values, 2)

    # 2. Suffix forward with cached prefix.
    suffix_logits = model(
        suffixes_flat, past_key_values=past_kv, use_cache=False
    ).logits  # [B*2, L-p, V]

    # 3. 4-nuc log-softmax — shared by LLR and JSD. fp32 cast inherits
    #    the biofoundation#21 numerical-stability fix.
    nuc_ids = nuc_token_ids.to(suffix_logits.device)
    log_p_nuc = F.log_softmax(
        suffix_logits[..., nuc_ids].float(), dim=-1
    )  # [B*2, L-p, 4]
    log_p_nuc = rearrange(log_p_nuc, "(B V) L C -> B V L C", B=B)  # [B, 2, L-p, 4]
    log_p_ref = log_p_nuc[:, 0, :-1]  # [B, L-p-1, 4] — drop last (predicts off-the-end)
    log_p_alt = log_p_nuc[:, 1, :-1]  # [B, L-p-1, 4]

    # 4. JSD over downstream positions (suffix indices [0, L-p-2] = global [p, L-2]).
    log_m = torch.logaddexp(log_p_ref, log_p_alt) - math.log(2.0)
    p_ref_dist = log_p_ref.exp()
    p_alt_dist = log_p_alt.exp()
    kl_ref_m = (p_ref_dist * (log_p_ref - log_m)).sum(dim=-1)  # [B, L-p-1]
    kl_alt_m = (p_alt_dist * (log_p_alt - log_m)).sum(dim=-1)
    next_token_jsd_mean = (0.5 * (kl_ref_m + kl_alt_m)).mean(dim=-1)  # [B]

    # 5. LLR = (variant-position contribution at p-1) + (downstream contribution at [p, L-2]).
    #    All in 4-nuc-softmax space; the log_full(nuc | context) terms cancel
    #    at the variant position and ≈ 0 elsewhere for trained DNA models.
    prefix_log_p = F.log_softmax(
        prefix_last_logits[..., nuc_ids].float(), dim=-1
    )  # [B, 4]
    ref_var_idx = _token_id_to_nuc_idx(input_ids[:, p], nuc_ids)  # [B]
    alt_var_idx = _token_id_to_nuc_idx(alt_token_id, nuc_ids)  # [B]
    llr_at_var = prefix_log_p.gather(-1, alt_var_idx.unsqueeze(-1)).squeeze(
        -1
    ) - prefix_log_p.gather(-1, ref_var_idx.unsqueeze(-1)).squeeze(-1)  # [B]

    # Downstream targets are shared between ref and alt (only var_pos differs).
    suffix_targets = input_ids[:, p + 1 :]  # [B, L-p-1]
    target_idx = _token_id_to_nuc_idx(suffix_targets, nuc_ids).unsqueeze(
        -1
    )  # [B, L-p-1, 1]
    log_p_ref_at_targets = log_p_ref.gather(-1, target_idx).squeeze(-1)  # [B, L-p-1]
    log_p_alt_at_targets = log_p_alt.gather(-1, target_idx).squeeze(-1)
    llr_downstream = (log_p_alt_at_targets - log_p_ref_at_targets).sum(dim=-1)  # [B]

    llr = llr_at_var + llr_downstream

    return torch.stack([llr, next_token_jsd_mean], dim=1)


def _token_id_to_nuc_idx(
    token_ids: Int[Tensor, "..."],
    nuc_token_ids: Int[Tensor, " 4"],
) -> Int[Tensor, "..."]:
    """Map a tensor of nucleotide token IDs to indices into ``nuc_token_ids``.

    Asserts every token is one of the four nucleotides (raises otherwise —
    catches non-SNV input that would silently miscompute downstream)."""
    eq = token_ids.unsqueeze(-1) == nuc_token_ids
    assert eq.any(dim=-1).all(), (
        "non-nucleotide token in SNV input — expected only ACGT token IDs"
    )
    return eq.int().argmax(dim=-1)


def _repeat_interleave_kv_cache(past_kv: Any, n: int) -> Any:
    """Repeat each layer's K and V along the batch dim by ``n``.

    Always returns an HF ``DynamicCache`` (constructing one from a legacy
    tuple if needed). Modern Qwen3/Llama-style models call
    ``past_key_values.get_seq_length()`` internally — the legacy
    tuple-of-(K, V)-pairs format normally auto-converts, but under
    ``torch.compile`` the conversion can be skipped and the method call
    raises ``AttributeError: 'tuple' object has no attribute
    'get_seq_length'``. Returning a real ``DynamicCache`` sidesteps that.

    Mutates an input ``Cache`` in place — caller doesn't reuse the original.
    """
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i in range(len(past_kv.key_cache)):
            past_kv.key_cache[i] = past_kv.key_cache[i].repeat_interleave(n, dim=0)
            past_kv.value_cache[i] = past_kv.value_cache[i].repeat_interleave(n, dim=0)
        return past_kv

    # Legacy tuple format → coerce to DynamicCache.
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_kv):
        new_cache.update(
            k.repeat_interleave(n, dim=0),
            v.repeat_interleave(n, dim=0),
            layer_idx=layer_idx,
        )
    return new_cache
