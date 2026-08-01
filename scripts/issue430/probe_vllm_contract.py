"""Check whether the issue #430 checkpoint can satisfy vLLM's score contract.

This is a metadata-only preflight. It does not load model weights or require a
GPU/vLLM installation. The GPU compatibility screen must still compare vLLM
prompt scores with the BF16 PyTorch reference before timing it.
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL = "marin-dna/marin-dna-exp135-m5.1"
DEFAULT_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    assert config.architectures == ["Qwen3ForCausalLM"], config.architectures
    assert config.model_type == "qwen3", config.model_type
    assert tokenizer.vocab_size == config.vocab_size

    nucleotide_token_ids: dict[str, int] = {}
    for nucleotide in "ACGT":
        token_ids = tokenizer.encode(nucleotide, add_special_tokens=False)
        assert len(token_ids) == 1, (nucleotide, token_ids)
        nucleotide_token_ids[nucleotide] = token_ids[0]

    assert len(set(nucleotide_token_ids.values())) == 4
    assert all(
        0 <= token_id < config.vocab_size for token_id in nucleotide_token_ids.values()
    )
    assert tokenizer.bos_token_id is not None

    result = {
        "model": args.model,
        "revision": args.revision,
        "architecture": config.architectures[0],
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "nucleotide_token_ids": nucleotide_token_ids,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "vllm_contract": {
            "prompt_logprobs": -1,
            "logprobs_mode": "raw_logprobs",
            "returned_values_per_scored_position": config.vocab_size,
            "normalization": "subtract logsumexp over token IDs for A,C,G,T",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
