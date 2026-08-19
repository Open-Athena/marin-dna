"""Pinned Qwen3 loading and `[MASK]` vocabulary expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from exp479_mntp.config import MASK_TOKEN, MODEL_ID, MODEL_REVISION

Initialization = Literal["transferred", "scratch"]
AttentionMode = Literal["causal", "full"]


@dataclass(frozen=True)
class ModelBundle:
    """A model/tokenizer pair plus vocabulary metadata."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    canonical_token_ids: tuple[int, ...]
    mask_token_id: int | None
    input_output_tied: bool


def canonical_token_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, ...]:
    """Return the A/C/G/T token IDs in canonical order."""

    token_ids = tuple(int(tokenizer.convert_tokens_to_ids(base)) for base in ("a", "c", "g", "t"))
    if len(set(token_ids)) != 4 or any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"tokenizer does not expose four distinct A/C/G/T tokens: {token_ids}")
    return token_ids


def add_mask_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[int, bool]:
    """Add `[MASK]` and initialize input/output rows from their A/C/G/T means.

    The released m5.1 config is untied. The implementation preserves that state
    and initializes the two new rows independently. If a future compatible
    checkpoint is tied, the shared matrix is initialized once.
    """

    old_vocab_size = int(model.get_input_embeddings().num_embeddings)
    added = tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    mask_token_id = int(tokenizer.mask_token_id)
    if mask_token_id < old_vocab_size and added != 0:
        raise RuntimeError("tokenizer reported an added token at an existing ID")

    input_before = model.get_input_embeddings().weight
    output_module_before = model.get_output_embeddings()
    output_before = None if output_module_before is None else output_module_before.weight
    was_tied = output_before is input_before

    if len(tokenizer) != old_vocab_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    output_weight = None if output_module is None else output_module.weight
    bases = canonical_token_ids(tokenizer)

    with torch.no_grad():
        input_weight[mask_token_id].copy_(input_weight[list(bases)].mean(dim=0))
        if output_weight is not None and output_weight is not input_weight:
            output_weight[mask_token_id].copy_(output_weight[list(bases)].mean(dim=0))

    model.config.vocab_size = len(tokenizer)
    model.config.mask_token_id = mask_token_id
    return mask_token_id, was_tied


def load_model_bundle(
    *,
    initialization: Initialization,
    add_mask: bool,
    attention_implementation: str = "sdpa",
    dtype: torch.dtype | None = None,
) -> ModelBundle:
    """Load transferred weights or construct the matched random architecture."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config.use_cache = False
    if initialization == "transferred":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            config=config,
            attn_implementation=attention_implementation,
            dtype=dtype,
        )
    elif initialization == "scratch":
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attention_implementation,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unknown initialization {initialization!r}")

    mask_token_id = None
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    output_weight = None if output_module is None else output_module.weight
    tied = output_weight is input_weight
    if add_mask:
        mask_token_id, tied = add_mask_token(model, tokenizer)

    model.config.use_cache = False
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        canonical_token_ids=canonical_token_ids(tokenizer),
        mask_token_id=mask_token_id,
        input_output_tied=tied,
    )


def model_logits(
    model: PreTrainedModel,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    attention_mode: AttentionMode,
) -> torch.Tensor:
    """Run Qwen3 with an explicit causal/full-attention choice."""

    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        is_causal=attention_mode == "causal",
        return_dict=True,
    )
    return outputs.logits
