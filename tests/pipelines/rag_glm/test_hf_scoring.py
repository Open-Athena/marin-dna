"""Focused tests for the issue #402 Hugging Face cached scorer."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from marin_dna.pipelines.rag_glm.hf_scoring import (
    RAG_COMPLETION_TOKENS,
    RAG_HUMAN_POOL_START,
    RAG_PREFIX_TOKENS,
    score_rag_completions_hf,
    score_rag_completions_naive_hf,
)


class _FakeCache:
    def __init__(self, histories: torch.Tensor):
        self.histories = histories

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.histories = self.histories.repeat_interleave(repeats, dim=0)


class _CountingCausalModel:
    """Tiny deterministic causal model with a duck-typed HF cache surface."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size
        self.calls: list[dict[str, object]] = []

    def _logits(self, histories: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([histories, tokens], dim=1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        absolute = positions.unsqueeze(0) + histories.shape[1]
        cumulative = combined.cumsum(dim=1)[:, histories.shape[1] :]
        vocab = torch.arange(self.vocab_size, device=tokens.device).view(1, 1, -1)
        centers = (cumulative + absolute) % self.vocab_size
        return -((vocab - centers.unsqueeze(-1)).float() ** 2) / 7.0

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        use_cache: bool,
        logits_to_keep: int | None = None,
        past_key_values: _FakeCache | None = None,
    ) -> SimpleNamespace:
        histories = (
            torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype)
            if past_key_values is None
            else past_key_values.histories
        )
        self.calls.append(
            {
                "shape": tuple(input_ids.shape),
                "has_cache": past_key_values is not None,
                "use_cache": use_cache,
            }
        )
        logits = self._logits(histories, input_ids)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
        cache = _FakeCache(torch.cat([histories, input_ids], dim=1))
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _naive_score(
    model: _CountingCausalModel,
    prefix: torch.Tensor,
    completion: torch.Tensor,
    nucleotide_ids: torch.Tensor,
) -> torch.Tensor:
    full = torch.cat([prefix, completion], dim=1)
    logits = model(full, use_cache=False).logits
    selected = logits[:, prefix.shape[1] - 1 : -1, nucleotide_ids]
    log_probs = F.log_softmax(selected.float(), dim=-1)
    matches = completion.unsqueeze(-1) == nucleotide_ids
    targets = matches.to(torch.int64).argmax(dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def test_cached_scores_match_naive_and_prefix_runs_once() -> None:
    nucleotide_ids = torch.tensor([4, 5, 6, 7])
    prefix = (torch.arange(2 * RAG_PREFIX_TOKENS).reshape(2, -1) % 4) + 4
    ref = (torch.arange(2 * RAG_COMPLETION_TOKENS).reshape(2, -1) % 4) + 4
    alt = ref.clone()
    alt[:, 0] = torch.tensor([5, 6])
    assert bool((alt[:, 0] != ref[:, 0]).all())

    cached_model = _CountingCausalModel()
    cached = score_rag_completions_hf(
        cached_model,
        prefix,
        ref,
        alt,
        nucleotide_token_ids=nucleotide_ids,
    )
    naive_paired = score_rag_completions_naive_hf(
        _CountingCausalModel(),
        prefix,
        ref,
        alt,
        nucleotide_token_ids=nucleotide_ids,
    )

    naive_model = _CountingCausalModel()
    naive_ref = _naive_score(naive_model, prefix, ref, nucleotide_ids)
    naive_alt = _naive_score(naive_model, prefix, alt, nucleotide_ids)

    torch.testing.assert_close(cached[:, 0], naive_ref)
    torch.testing.assert_close(cached[:, 1], naive_alt)
    torch.testing.assert_close(cached, naive_paired, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(cached[:, 2], cached[:, 1] - cached[:, 0])
    torch.testing.assert_close(
        cached[:, 2], naive_alt - naive_ref, rtol=1e-4, atol=5e-5
    )
    assert cached_model.calls == [
        {"shape": (2, RAG_PREFIX_TOKENS), "has_cache": False, "use_cache": True},
        {
            "shape": (4, RAG_COMPLETION_TOKENS),
            "has_cache": True,
            "use_cache": False,
        },
    ]


def test_rejects_non_shared_downstream_completion() -> None:
    prefix = torch.full((1, RAG_PREFIX_TOKENS), 4)
    ref = torch.full((1, RAG_COMPLETION_TOKENS), 4)
    alt = ref.clone()
    alt[:, 0] = 5
    alt[:, 1] = 6

    try:
        score_rag_completions_hf(
            _CountingCausalModel(),
            prefix,
            ref,
            alt,
            nucleotide_token_ids=torch.tensor([4, 5, 6, 7]),
        )
    except AssertionError as exc:
        assert "differ only at the SNV" in str(exc)
    else:  # pragma: no cover - fail loudly without adding pytest to the module
        raise AssertionError("expected malformed paired completions to fail")


def test_literal_human_only_cached_scores_match_naive() -> None:
    nucleotide_ids = torch.tensor([4, 5, 6, 7])
    prefix_tokens = 128
    prefix = torch.cat(
        [torch.tensor([[2]]), torch.full((1, prefix_tokens - 1), 4)], dim=1
    )
    ref = torch.full((1, RAG_COMPLETION_TOKENS), 5)
    alt = ref.clone()
    alt[:, 0] = 6
    cached = score_rag_completions_hf(
        _CountingCausalModel(),
        prefix,
        ref,
        alt,
        nucleotide_token_ids=nucleotide_ids,
        expected_prefix_tokens=prefix_tokens,
    )
    naive = score_rag_completions_naive_hf(
        _CountingCausalModel(),
        prefix,
        ref,
        alt,
        nucleotide_token_ids=nucleotide_ids,
        expected_prefix_tokens=prefix_tokens,
    )
    torch.testing.assert_close(cached, naive, rtol=1e-4, atol=5e-5)


def test_human_segment_embeddings_match_naive_full_forwards() -> None:
    torch.manual_seed(402)
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2_048,
        bos_token_id=2,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = Qwen3ForCausalLM(config).eval()
    prefix = torch.cat(
        [torch.tensor([[2]]), torch.randint(4, 8, (1, RAG_PREFIX_TOKENS - 1))],
        dim=1,
    )
    ref = torch.randint(4, 8, (1, RAG_COMPLETION_TOKENS))
    alt = ref.clone()
    alt[:, 0] = ((ref[:, 0] - 4 + 1) % 4) + 4
    nucleotide_ids = torch.tensor([4, 5, 6, 7])

    with torch.no_grad():
        scores_only = score_rag_completions_hf(
            model,
            prefix,
            ref,
            alt,
            nucleotide_token_ids=nucleotide_ids,
        )
        scores_and_embeddings = score_rag_completions_hf(
            model,
            prefix,
            ref,
            alt,
            nucleotide_token_ids=nucleotide_ids,
            return_embeddings=True,
        )
        ref_hidden = model(
            torch.cat([prefix, ref], dim=1), output_hidden_states=True
        ).hidden_states[-1]
        alt_hidden = model(
            torch.cat([prefix, alt], dim=1), output_hidden_states=True
        ).hidden_states[-1]

    hidden_size = config.hidden_size
    assert scores_and_embeddings.shape == (1, 3 + 2 * hidden_size)
    torch.testing.assert_close(scores_and_embeddings[:, :3], scores_only)
    torch.testing.assert_close(
        scores_and_embeddings[:, 3 : 3 + hidden_size],
        ref_hidden[:, RAG_HUMAN_POOL_START:].float().mean(dim=1),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        scores_and_embeddings[:, 3 + hidden_size :],
        alt_hidden[:, RAG_HUMAN_POOL_START:].float().mean(dim=1),
        rtol=1e-5,
        atol=1e-5,
    )
    assert not model.base_model._forward_hooks
