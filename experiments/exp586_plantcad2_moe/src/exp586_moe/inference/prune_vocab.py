"""Create an exact compact-vocabulary HF export for ACGTN-only inference."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from exp586_moe.inference.tokenizer import compact_dna_tokenizer

EMBEDDING_WEIGHT = "model.embed_tokens.weight"
OUTPUT_WEIGHT = "lm_head.weight"
COMPACT_TOKENS = "acgtn"


def _trained_ids(source: Tokenizer) -> list[int]:
    ids = [source.token_to_id(base) for base in COMPACT_TOKENS]
    if any(token_id is None for token_id in ids):
        raise ValueError("Source tokenizer lacks an isolated ACGTN token")
    return [int(token_id) for token_id in ids]


def compact_hf_export(source: Path, output: Path) -> None:
    """Remap the five reachable embedding/head rows while preserving all other weights."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    original_ids = _trained_ids(tokenizer)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    for name in (EMBEDDING_WEIGHT, OUTPUT_WEIGHT):
        if name not in weight_map:
            raise ValueError(f"Export lacks {name}")

    total_size = 0
    for shard in sorted(set(weight_map.values())):
        source_shard = source / shard
        tensors = load_file(source_shard, device="cpu")
        for name in (EMBEDDING_WEIGHT, OUTPUT_WEIGHT):
            if name in tensors:
                tensors[name] = tensors[name][original_ids].contiguous()
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        save_file(tensors, temporary / shard, metadata=metadata)
        total_size += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )

    index.setdefault("metadata", {})["total_size"] = total_size
    (temporary / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    config = json.loads((source / "config.json").read_text())
    config["vocab_size"] = len(COMPACT_TOKENS)
    config["exp586_compact_vocab"] = {
        "tokens": COMPACT_TOKENS,
        "original_ids": original_ids,
    }
    (temporary / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    compact_dna_tokenizer().save(str(temporary / "tokenizer.json"))
    shutil.copy2(source / "tokenizer_config.json", temporary / "tokenizer_config.json")
    temporary.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    compact_hf_export(arguments.source, arguments.output)
