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

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.cache_utils import DynamicCache

from marin_dna.data.dna import COMPLEMENT, NUCLEOTIDES


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
    return_embeddings: bool = False,
    pool_lo: int | None = None,
    pool_hi: int | None = None,
) -> Float[Tensor, "B 2"] | Float[Tensor, "B 2+2D"]:
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
        return_embeddings: If True, also emit the entire-window mean-pooled
            last-layer hidden states for ref and alt — produced by requesting
            ``output_hidden_states=True`` on the **same two** (prefix, suffix)
            forwards (no extra pass). The prefix hidden states ``[B, p, D]`` are
            shared by ref/alt (causal, shared prefix); concatenated with each
            allele's suffix states ``[B, L-p, D]`` they equal a full forward's
            last-layer states. Pooling accumulates in fp32. Off by default —
            materializing ``[·, ·, D]`` hidden states is paid only when on.
        pool_lo: Token index (inclusive) where the pooled window starts. Required
            when ``return_embeddings``. The runner sets it to ``n_prefix`` so the
            BOS/special prefix tokens are **excluded** — the pool covers exactly
            the ``window_size`` DNA positions (matching the #314 ``entire_window``
            extent and ``compute_window_embedding``'s ``n_prefix`` offset).
        pool_hi: Token index (exclusive) where the pooled window ends. Required
            when ``return_embeddings``; the runner sets it to ``n_prefix +
            window_size``. ``pool_hi - pool_lo == window_size`` positions are
            pooled.

    Returns:
        When ``return_embeddings=False``, a tensor with shape ``[B, 2]``:
            - [:, 0]: LLR (alt_logprob - ref_logprob, 4-nuc-softmax space)
            - [:, 1]: next_token_jsd_mean (mean per-position 4-nuc JSD over downstream positions)
        When ``return_embeddings=True``, the scores are concatenated with the two
        pooled fp32 embeddings into ``[B, 2 + 2D]`` (``D`` = model hidden size):
        columns ``[0:2]`` are LLR/JSD as above, ``[2:2+D]`` is ``emb_ref``,
        ``[2+D:2+2D]`` is ``emb_alt``. A single tensor (not a tuple) keeps the
        HF ``Trainer.predict`` collation contract; the runner/driver slice it.
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
    #    output_hidden_states (only when pooling embeddings) keeps the full
    #    prefix hidden states [B, p, D] — logits_to_keep slices the lm_head, not
    #    the hidden states.
    prefix_out = model(
        prefix,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=return_embeddings,
    )
    prefix_last_logits = prefix_out.logits[:, -1]  # [B, V]
    past_kv = _repeat_interleave_kv_cache(prefix_out.past_key_values, 2)

    # 2. Suffix forward with cached prefix.
    suffix_out = model(
        suffixes_flat,
        past_key_values=past_kv,
        use_cache=False,
        output_hidden_states=return_embeddings,
    )
    suffix_logits = suffix_out.logits  # [B*2, L-p, V]

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

    scores = torch.stack([llr, next_token_jsd_mean], dim=1)  # [B, 2]
    if not return_embeddings:
        return scores

    # 6. Entire-window mean-pool of the last-layer hidden states, ref & alt.
    #    Pool over token positions [pool_lo, pool_hi) = the window_size DNA
    #    positions (n_prefix BOS tokens excluded). The prefix forward gives the
    #    shared states [B, p, D]; the suffix forward (run with the cached prefix)
    #    gives each allele's states at [p, L), so prefix[pool_lo:] ++ suffix[:pool_hi-p]
    #    reconstructs the pooled-window states without a full re-forward. The
    #    var-position token is the suffix's first position — included in the pool.
    #    Accumulate in fp32 (bf16/f16 summed over hundreds of positions accrues
    #    rounding error); the driver FWD+RC-averages (still fp32) and casts to f16.
    assert pool_lo is not None and pool_hi is not None, (
        "pool_lo/pool_hi are required when return_embeddings=True"
    )
    assert 0 <= pool_lo < p < pool_hi <= L, (
        f"pool bounds [{pool_lo}, {pool_hi}) invalid for var_pos {p}, length {L}; "
        f"expected 0 <= pool_lo < var_pos < pool_hi <= L"
    )
    prefix_hidden = prefix_out.hidden_states[-1]  # [B, p, D]
    suffix_hidden = rearrange(
        suffix_out.hidden_states[-1], "(B V) S D -> B V S D", B=B
    )  # [B, 2, L-p, D]
    n_pool = pool_hi - pool_lo
    prefix_sum = prefix_hidden[:, pool_lo:].float().sum(dim=1)  # [B, D] (shared)
    ref_sum = prefix_sum + suffix_hidden[:, 0, : pool_hi - p].float().sum(dim=1)
    alt_sum = prefix_sum + suffix_hidden[:, 1, : pool_hi - p].float().sum(dim=1)
    emb_ref = ref_sum / n_pool  # [B, D]
    emb_alt = alt_sum / n_pool  # [B, D]
    return torch.cat([scores, emb_ref, emb_alt], dim=1)  # [B, 2 + 2D]


def compute_marginal_clm(
    model: Any,
    input_ids: Int[Tensor, "B L"],
    *,
    var_pos: int,
    nuc_token_ids: Int[Tensor, " 4"],
) -> Float[Tensor, "B 4"]:
    """Per-site 4-allele marginal ``log p(x)`` over ``{A,C,G,T}`` via prefix-sharing.

    The opt-in entropy/calibration atom (issue #269). Generalizes
    ``compute_variant_score_bundle``'s prefix-sharing kernel from 2 alleles
    (ref/alt) to all 4: the four alleles share the prefix ``input_ids[:var_pos]``
    and the downstream tail ``input_ids[var_pos+1:]``, differing only at the
    variant token, so the shared prefix is forwarded once (KV-cached) and the
    four divergent suffixes (length ``L - var_pos``) are forwarded against it —
    1 prefix + 4 suffix forwards, vs the 4 full-length forwards of the legacy
    ``compute_reflogprob_clm``.

    Works in the same **4-nucleotide softmax** space as the bundle. For allele
    ``x`` the full-sequence score, up to the shared-prefix log-prob (constant
    across alleles → cancels in the final softmax), is

        ``L(x) = log p(x | prefix) + Σ_t log p(tail_t | prefix, x, tail_<t)``

    and the marginal is ``log_softmax_x L(x)``. By construction
    ``marginal[:, alt] − marginal[:, ref]`` is *identically* the bundle's LLR
    (the softmax normalizer cancels in the difference), so calibration LLRs
    derived here match the eval LLRs exactly.

    Downstream CPU reductions on the returned ``[B, 4]`` give the per-site
    Shannon entropy (``entropy_from_marginal``), the ref log-prob
    (``marginal[:, ref]``), and all three LLRs (``marginal[:, alt] −
    marginal[:, ref]``) with no further forward passes.

    Args:
        model: HF-shaped causal LM (see ``compute_variant_score_bundle`` for the
            duck-typed cache / ``logits_to_keep`` contract).
        input_ids: Reference window token IDs, shape ``[B, L]``. Only the prefix
            ``[:var_pos]`` and tail ``[var_pos+1:]`` are read; the token at
            ``var_pos`` is overwritten by each allele.
        var_pos: Token-level variant position (Python int, constant within the
            batch — derived in the runner from ``window_size`` / strand /
            ``n_prefix`` and bound via ``partial`` so it never graph-breaks
            torch.compile).
        nuc_token_ids: Length-4 tensor of token IDs for A/C/G/T in
            ``NUCLEOTIDES`` order. The returned columns follow this order.

    Returns:
        ``[B, 4]`` marginal log-probabilities ``log p(x)`` over the four
        nucleotides at ``var_pos`` (each row ``logsumexp``-es to 0).
    """
    B, L = input_ids.shape
    p = var_pos
    assert 0 < p < L, (
        f"variant at token position {p} of length-{L} sequence has no shared "
        f"prefix; expected 0 < var_pos < L"
    )

    nuc = nuc_token_ids.to(input_ids.device)  # [4]
    n_alleles = nuc.shape[0]

    # The 4 alleles differ only at var_pos; build each allele's suffix as
    # [allele_x] + shared_tail (tail = input_ids[:, p+1:], identical across
    # alleles). [B, V, L-p] flattened "(B V)" matches _repeat_interleave_kv_cache's
    # repeat_interleave ordering — exactly how compute_variant_score_bundle lays
    # out its 2 alleles, generalized to 4.
    tail = input_ids[:, p + 1 :]  # [B, L-p-1] (empty iff p == L-1)
    allele_tokens = nuc.view(1, n_alleles, 1).expand(B, n_alleles, 1)  # [B, 4, 1]
    tail_rep = tail.unsqueeze(1).expand(B, n_alleles, tail.shape[1])  # [B, 4, L-p-1]
    suffixes = torch.cat([allele_tokens, tail_rep], dim=-1)  # [B, 4, L-p]
    suffixes_flat = rearrange(suffixes, "B V L -> (B V) L").contiguous()  # [B*4, L-p]

    # 1. Prefix forward — only the last position (predicts the variant token).
    prefix = input_ids[:, :p].contiguous()
    prefix_out = model(prefix, use_cache=True, logits_to_keep=1)
    prefix_last_logits = prefix_out.logits[:, -1]  # [B, V]
    past_kv = _repeat_interleave_kv_cache(prefix_out.past_key_values, n_alleles)

    # 2. Suffix forward with the cached prefix.
    suffix_logits = model(
        suffixes_flat, past_key_values=past_kv, use_cache=False
    ).logits  # [B*4, L-p, V]

    # 3. 4-nuc log-softmax (fp32 — inherits the biofoundation#21 stability fix).
    nuc_ids = nuc.to(suffix_logits.device)
    log_p_nuc = F.log_softmax(
        suffix_logits[..., nuc_ids].float(), dim=-1
    )  # [B*4, L-p, 4]
    log_p_nuc = rearrange(log_p_nuc, "(B V) L C -> B V L C", B=B)  # [B, 4, L-p, 4]

    # 4. Variant-position term: log p(allele | prefix), [B, 4].
    prefix_log_p = F.log_softmax(
        prefix_last_logits[..., nuc_ids].float(), dim=-1
    )  # [B, 4]

    # 5. Downstream term: Σ_t log p(tail_t | prefix, allele, tail_<t), per allele.
    #    Drop the last suffix position (predicts off-the-end); the remaining
    #    L-p-1 positions predict the shared tail (same targets for every allele).
    log_p_down = log_p_nuc[:, :, :-1, :]  # [B, 4, L-p-1, 4]
    target_idx = _token_id_to_nuc_idx(tail, nuc_ids)  # [B, L-p-1]
    target_idx = target_idx.view(B, 1, -1, 1).expand(B, n_alleles, -1, 1)
    down_term = log_p_down.gather(-1, target_idx).squeeze(-1).sum(dim=-1)  # [B, 4]

    # 6. Marginal: the shared-prefix log-prob is constant across alleles and
    #    cancels in the softmax, so log_softmax(var-term + downstream-term) is
    #    the full-sequence 4-allele marginal.
    marginal_log_prob = F.log_softmax(prefix_log_p + down_term, dim=-1)  # [B, 4]
    return marginal_log_prob


def entropy_from_marginal(marginal_log_prob: np.ndarray) -> np.ndarray:
    """Shannon entropy ``H = −Σ_x p(x)·log p(x)`` of a 4-allele marginal.

    Pure-CPU reduction over the last axis of a ``[..., 4]`` array — the
    per-strand output of ``compute_marginal_clm`` /
    ``marin_dna.model.runner.run_variant_marginal``, or an ``rc_average_marginal``
    of the two strands. Returns ``[...]`` entropy in nats, in ``[0, log 4]``
    (``log 4`` at the uniform marginal, ``→ 0`` at a degenerate one).

    **Normalizes its input internally** (``log_softmax`` over the 4 alleles), so
    it accepts either a normalized log-marginal or an unnormalized one — logits
    up to a per-row additive constant. That is what lets the FWD+RC combination
    be a plain mean of log-probs with no separate renormalization step
    (``rc_average_marginal``): the normalization is deferred to here. Idempotent
    on an already-normalized marginal. Accepts NumPy arrays, the dtype the HF
    Trainer harness returns.
    """
    logp = np.asarray(marginal_log_prob, dtype=np.float64)
    logp = logp - np.logaddexp.reduce(logp, axis=-1, keepdims=True)  # log_softmax
    return -(np.exp(logp) * logp).sum(axis=-1)


# RC marginal columns are the complement alleles: column i of the RC-strand
# marginal is the forward-strand allele complement(NUCLEOTIDES[i]). Realigning to
# forward-strand ACGT order before averaging is the A↔T / C↔G reversal [3,2,1,0].
_RC_ALLELE_PERM = [NUCLEOTIDES.index(COMPLEMENT[n]) for n in NUCLEOTIDES]


def rc_average_marginal(fwd: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """FWD+RC mean of two per-strand 4-allele log-marginals, in forward ACGT order.

    The forward and reverse-complement passes of ``run_variant_marginal`` estimate
    the same site distribution from opposite strands. This averages their
    log-probabilities (the geometric mean of the two strand distributions) after
    realigning the RC columns to forward-strand allele order — the RC marginal's
    column ``i`` is the forward allele ``complement(NUCLEOTIDES[i])``, so the
    realignment is the ``A↔T / C↔G`` reversal ``[3, 2, 1, 0]``. A naive
    ``(fwd + rc) / 2`` would average mismatched alleles: a silent strand bug.

    Returns the averaged log-marginal ``[..., 4]`` (forward ACGT order), left
    **unnormalized** — feed it to ``entropy_from_marginal`` (which normalizes
    internally) or take allele differences for the LLR (normalization-invariant).
    The average is linear in the log-marginal, so ``avg[alt] − avg[ref]`` equals
    the mean of the two per-strand LLRs — matching how the eval bundle FWD+RC
    -averages.
    """
    fwd = np.asarray(fwd, dtype=np.float64)
    rc = np.asarray(rc, dtype=np.float64)
    return 0.5 * (fwd + rc[..., _RC_ALLELE_PERM])


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


def compute_window_embedding(
    model: Any,
    input_ids: Int[Tensor, "B L"],
    *,
    tok_lo: int,
    tok_hi: int,
    layer_index: int = -1,
) -> Float[Tensor, "B D"]:
    """Mean-pool one layer's hidden state over the center tokens ``[tok_lo, tok_hi)``.

    The embedding-UMAP readout (issue #246). For the last layer
    (``layer_index == -1``) reads ``last_hidden_state`` directly — so ``model``
    should be a base ``AutoModel`` (no LM head to waste compute/memory on). For
    an intermediate layer reads ``hidden_states[layer_index]`` (the forward must
    accept ``output_hidden_states=True``). Pools in fp32 to avoid bf16
    accumulation error across the pooled positions (cf. the log_softmax fp32
    casts above). Returns ``[B, D]`` (``D`` = model hidden size).
    """
    if layer_index == -1:
        hidden = model(input_ids).last_hidden_state  # [B, L, D]
    else:
        hidden = model(input_ids, output_hidden_states=True).hidden_states[layer_index]
    # Defensive: a tokenizer that adds an unexpected suffix, or a bad
    # bounds calc, would otherwise silently truncate the pool.
    assert 0 <= tok_lo < tok_hi <= hidden.shape[1], (
        f"center slice [{tok_lo}:{tok_hi}] out of bounds for seq length {hidden.shape[1]}"
    )
    return hidden[:, tok_lo:tok_hi].float().mean(dim=1)  # [B, D]
