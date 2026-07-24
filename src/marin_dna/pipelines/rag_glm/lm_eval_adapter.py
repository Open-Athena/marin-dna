"""Current-Marin lm-eval adapter for paired issue #402 RAG scoring.

The request deliberately carries ``context``, ``ref_completion``, and
``alt_completion`` together.  The Levanter method executes the shared prefix
once and branches only after the page-aligned 1,920-token prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marin_dna.pipelines.rag_glm.hf_scoring import (
    RAG_COMPLETION_TOKENS,
    RAG_PREFIX_TOKENS,
)


@dataclass(frozen=True)
class EncodedRagRequest:
    prefix_ids: tuple[int, ...]
    ref_completion_ids: tuple[int, ...]
    alt_completion_ids: tuple[int, ...]
    nucleotide_token_ids: tuple[int, ...]


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if encoded and isinstance(encoded[0], list):
        assert len(encoded) == 1
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def encode_rag_request(
    tokenizer: Any,
    context: str,
    ref_completion: str,
    alt_completion: str,
) -> EncodedRagRequest:
    """Tokenize one frozen materialized row and enforce its exact geometry."""
    assert context.count("[SEQ]") == 7
    bos_token_id = tokenizer.bos_token_id
    assert bos_token_id is not None, "RAG scoring requires the document BOS/CLS token"
    prefix_ids = [
        int(bos_token_id),
        *_encode_without_special_tokens(tokenizer, context),
    ]
    ref_ids = _encode_without_special_tokens(tokenizer, ref_completion)
    alt_ids = _encode_without_special_tokens(tokenizer, alt_completion)
    nucleotide_ids = tuple(
        _encode_without_special_tokens(tokenizer, nucleotide)[0]
        for nucleotide in "ACGT"
    )
    assert len(set(nucleotide_ids)) == 4
    assert len(prefix_ids) == RAG_PREFIX_TOKENS
    assert len(ref_ids) == RAG_COMPLETION_TOKENS
    assert len(alt_ids) == RAG_COMPLETION_TOKENS
    assert ref_ids[0] != alt_ids[0]
    assert ref_ids[1:] == alt_ids[1:]
    nucleotide_id_set = set(nucleotide_ids)
    assert set(ref_ids) <= nucleotide_id_set
    assert set(alt_ids) <= nucleotide_id_set
    return EncodedRagRequest(
        prefix_ids=tuple(prefix_ids),
        ref_completion_ids=tuple(ref_ids),
        alt_completion_ids=tuple(alt_ids),
        nucleotide_token_ids=nucleotide_ids,
    )


def install_levanter_rag_loglikelihood() -> None:
    """Install the isolated ``rag_loglikelihood`` method on current Marin.

    The exp402 training resource is one v5p-8 host.  Multi-host dispatch would
    require extending Marin's private worker protocol, so fail explicitly if a
    later recipe changes that invariant.
    """
    try:
        import haliax as hax
        import jax
        import jax.numpy as jnp
        from levanter.eval_harness import LevanterHarnessLM
    except ImportError:
        return

    if getattr(LevanterHarnessLM, "_marin_dna_rag_patched", False):
        return

    from marin_dna.pipelines.rag_glm.levanter_scoring import (
        score_rag_completions_levanter,
    )

    def rag_loglikelihood(
        self: Any, requests: list[Any]
    ) -> list[tuple[float, float, float]]:
        assert jax.process_count() == 1, (
            "issue #402's paged-cache lm-eval adapter supports one JAX host; "
            "keep the experiment on v5p-8 or add an explicit worker message"
        )

        if not hasattr(self, "_marin_dna_rag_jit"):
            mixed_precision = self.leader.mp

            def _score(
                model: Any, prefix: Any, ref: Any, alt: Any, nucleotides: Any
            ) -> Any:
                if mixed_precision is not None:
                    model = mixed_precision.cast_to_compute(model)
                return score_rag_completions_levanter(
                    model,
                    prefix,
                    ref,
                    alt,
                    nucleotides,
                )

            self._marin_dna_rag_jit = hax.named_jit(
                _score,
                axis_resources=self.axis_resources,
                out_axis_resources={},
            )

        outputs: list[tuple[float, float, float]] = []
        current_task = getattr(self, "_current_task", "rag_loglikelihood_task")
        for request in requests:
            context, ref_completion, alt_completion = request.args
            encoded = encode_rag_request(
                self.tokenizer,
                context,
                ref_completion,
                alt_completion,
            )
            bucket = self._prepare_bucket(current_task)
            if bucket is not None:
                bucket.append(
                    {
                        "prompt": context,
                        "generation": f"ref={ref_completion};alt={alt_completion}",
                    }
                )
            self._handle_profiler_step()
            scores = self._marin_dna_rag_jit(
                self.leader.model,
                jnp.asarray(encoded.prefix_ids, dtype=jnp.int32),
                jnp.asarray(encoded.ref_completion_ids, dtype=jnp.int32),
                jnp.asarray(encoded.alt_completion_ids, dtype=jnp.int32),
                jnp.asarray(encoded.nucleotide_token_ids, dtype=jnp.int32),
            )
            observed = jax.device_get(scores)
            assert observed.shape == (3,)
            outputs.append(tuple(float(value) for value in observed))
            self._current_step += 1
        self._stop_profiler_if_needed()
        return outputs

    LevanterHarnessLM.rag_loglikelihood = rag_loglikelihood
    LevanterHarnessLM._marin_dna_rag_patched = True
