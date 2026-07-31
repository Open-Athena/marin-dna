"""Exercise the real five-way m5.1 stream through pinned SAELens.

This uses the tiny random Qwen3 model from ``saelens_compatibility.py``;
there is no m5.1 checkpoint download and no accelerator launch.  Run from the
repository root with the same commit-pinned SAELens dependency::

    uv run --with \
      'sae-lens @ git+https://github.com/decoderesearch/SAELens@8be14080485952f729ed58d674bcddf9778e0aa4' \
      python scripts/issue288/live_saelens_data_smoke.py
"""

from __future__ import annotations

import json

import torch
from sae_lens.load_model import HookedProxyLM
from sae_lens.training.activations_store import ActivationsStore
from transformers import AutoTokenizer

from five_way_data import (
    CONTEXT_TOKENS,
    SOURCES,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    WINDOW_BP,
    build_five_way_dataset,
)
from saelens_compatibility import (
    HIDDEN_SIZE,
    HOOK_NAME,
    make_model,
)

BATCH_WINDOWS = len(SOURCES)


@torch.no_grad()
def run_live_smoke() -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
    )
    bos_token_id = tokenizer.bos_token_id
    assert bos_token_id is not None

    dataset = build_five_way_dataset(tokenizer)
    preview = list(dataset.take(BATCH_WINDOWS))
    expected_order = [source.name for source in SOURCES]
    assert [row["source"] for row in preview] == expected_order
    assert all(len(row["input_ids"]) == CONTEXT_TOKENS for row in preview)

    model = HookedProxyLM(
        make_model(),
        tokenizer,
        hook_names=[HOOK_NAME],
    )
    store = ActivationsStore(
        model=model,
        dataset=dataset,
        streaming=True,
        hook_name=HOOK_NAME,
        hook_head_index=None,
        context_size=CONTEXT_TOKENS,
        d_in=HIDDEN_SIZE,
        n_batches_in_buffer=1,
        total_training_tokens=BATCH_WINDOWS * WINDOW_BP,
        store_batch_size_prompts=BATCH_WINDOWS,
        train_batch_size_tokens=BATCH_WINDOWS * WINDOW_BP,
        prepend_bos=True,
        normalize_activations="none",
        device=torch.device("cpu"),
        dtype="float32",
        cached_activations_path=None,
        model_kwargs=None,
        autocast_lm=False,
        dataset_trust_remote_code=False,
        seqpos_slice=(None,),
        exclude_special_tokens=torch.tensor([bos_token_id], dtype=torch.long),
        disable_concat_sequences=True,
        sequence_separator_token="bos",
        activations_mixing_fraction=0.0,
        use_chat_formatting=False,
    )
    activations = store.get_filtered_llm_batch()
    expected_shape = (BATCH_WINDOWS * WINDOW_BP, HIDDEN_SIZE)
    assert tuple(activations.shape) == expected_shape
    assert torch.isfinite(activations).all()

    return {
        "source_order": expected_order,
        "input_shape": [BATCH_WINDOWS, CONTEXT_TOKENS],
        "bos_tokens_removed": BATCH_WINDOWS,
        "activation_shape": list(activations.shape),
        "activations_are_finite": True,
        "first_records": {row["source"]: row["record_id"] for row in preview},
    }


if __name__ == "__main__":
    print(json.dumps(run_live_smoke(), indent=2, sort_keys=True))
