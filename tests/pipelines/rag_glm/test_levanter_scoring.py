"""Focused tests for the issue #402 Levanter paged-KV scorer."""

from __future__ import annotations

import pytest

from marin_dna.pipelines.rag_glm.levanter_scoring import (
    build_rag_cache_execution_plan,
)


def test_cache_execution_plan_has_one_prefix_and_two_branches() -> None:
    plan = build_rag_cache_execution_plan(
        prefix_length=8, completion_length=4, page_size=4
    )
    assert plan.prefix_positions == tuple(range(8))
    assert set(plan.prefix_slots) == {0}
    assert plan.continuation_tokens_per_branch == 3
    assert plan.continuation_positions == (8, 9, 10, 8, 9, 10)
    assert plan.continuation_slots == (0, 0, 0, 1, 1, 1)


def test_online_cached_scores_match_naive_full_forward() -> None:
    pytest.importorskip("levanter.inference.page_table")
    hax = pytest.importorskip("haliax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from levanter.layers.attention import AttentionMask
    from levanter.models.qwen import Qwen3Config

    from marin_dna.pipelines.rag_glm.levanter_scoring import (
        score_rag_completions_levanter,
    )

    config = Qwen3Config(
        max_seq_len=12,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        use_sliding_window=False,
        tie_word_embeddings=False,
    )
    model = config.build(hax.Axis("vocab", 8), key=jax.random.key(402))
    prefix = jnp.asarray([2, 4, 5, 6, 7, 4, 5, 6], dtype=jnp.int32)
    reference = jnp.asarray([4, 5, 6, 7], dtype=jnp.int32)
    alternate = jnp.asarray([7, 5, 6, 7], dtype=jnp.int32)
    nucleotides = jnp.asarray([4, 5, 6, 7], dtype=jnp.int32)

    cached = score_rag_completions_levanter(
        model,
        prefix,
        reference,
        alternate,
        nucleotides,
        prefix_length=8,
        completion_length=4,
        page_size=4,
        compute_dtype=jnp.float32,
    )

    def naive(completion):
        tokens = jnp.concatenate([prefix, completion])
        Pos = hax.Axis("position", 12)
        named_tokens = hax.named(tokens, Pos)
        pos_ids = hax.named(jnp.arange(12, dtype=jnp.int32), Pos)
        logits = (
            model(
                named_tokens,
                attn_mask=AttentionMask.causal(),
                pos_ids=pos_ids,
                key=None,
            )
            .rearrange((Pos, model.Vocab))
            .array
        )
        selected = logits[7:11, :][:, nucleotides]
        log_probs = jax.nn.log_softmax(selected.astype(jnp.float32), axis=-1)
        targets = jnp.argmax(completion[:, None] == nucleotides, axis=-1)
        return jnp.take_along_axis(log_probs, targets[:, None], axis=-1).sum()

    naive_ref = naive(reference)
    naive_alt = naive(alternate)
    assert float(cached[0]) == pytest.approx(float(naive_ref), abs=2e-5)
    assert float(cached[1]) == pytest.approx(float(naive_alt), abs=2e-5)
    assert float(cached[2]) == pytest.approx(float(naive_alt - naive_ref), abs=2e-5)

    from marin_dna.pipelines.rag_glm.levanter_scoring import score_rag_batch_levanter

    batched = score_rag_batch_levanter(
        model,
        jnp.stack([prefix, prefix]),
        jnp.stack([reference, alternate]),
        jnp.stack([alternate, reference]),
        nucleotides,
        prefix_length=8,
        completion_length=4,
        page_size=4,
        compute_dtype=jnp.float32,
    )
    assert batched.shape == (2, 3)
    assert float(batched[0, 0]) == pytest.approx(float(naive_ref), abs=2e-5)
    assert float(batched[0, 1]) == pytest.approx(float(naive_alt), abs=2e-5)
