#!/usr/bin/env python3
"""Normalize one staged issue #402 checkpoint before Hugging Face upload."""

from __future__ import annotations

import argparse
from pathlib import Path

from marin_dna_evals.rag_glm.hf_publication import normalize_rag_hf_export_metadata
from marin_dna_evals.rag_glm.offline_eval import (
    load_rag_model_config_hf,
    load_rag_tokenizer_hf,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    normalize_rag_hf_export_metadata(args.model_dir)
    tokenizer = load_rag_tokenizer_hf(args.model_dir)
    config = load_rag_model_config_hf(args.model_dir)
    assert tokenizer.model_max_length == 2_048
    assert config.max_position_embeddings == 2_048
    rope_parameters = getattr(config, "rope_parameters", {})
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope_parameters.get("rope_theta")
    assert rope_theta == 500_000
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = rope_parameters
    assert rope_scaling["rope_type"] == "llama3"
    print(f"Normalized and validated HF metadata in {args.model_dir}")


if __name__ == "__main__":
    main()
