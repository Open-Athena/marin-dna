"""Throwaway example of the restricted Torch baseline, using the experiment code."""

import argparse
import os
from pathlib import Path

os.environ["TORCHINDUCTOR_FREEZING"] = "1"  # Set before importing Torch.

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from exp586_moe.inference.contract import token_contract
from exp586_moe.inference.tokenizer import (
    compact_dna_unigram_tokenizer,
    encode_dna_batch,
)
from exp586_moe.inference.torch_backend import EagerGrugMoE


def main(model_dir: Path | None = None) -> None:
    """Produce all four base logits for one complete 8,192-base window."""
    if model_dir is None:
        model_dir = Path(
            snapshot_download(
                "plantcad/marin-dna-exp586-1.5B-60M-216B",
                revision="4130f71edcd117a7375af1ac78723d9a9173533f",
                allow_patterns=["*.json", "*.safetensors"],
            )
        )
    device = torch.device("cuda:0")
    contract = token_contract(Tokenizer.from_file(str(model_dir / "tokenizer.json")))
    base_ids = [contract.output_ids[base] for base in "ACGT"]
    model = EagerGrugMoE(
        model_dir,
        device,
        attention_backend="vllm_fa3",  # Existing adapter name; version selects FA2.
        flash_attention_version=2,
        compact_input_ids=[contract.input_ids[base] for base in "ACGTN"],
        compact_output_ids=base_ids,
        specialized_topk=False,
        sconv_backend="direct",
        precomputed_rope=True,
        combined_shared_expert_projections=True,
        prepare_vllm_moe=False,
    )

    def forward(ids: torch.Tensor) -> torch.Tensor:
        return model(ids, base_ids, variant="acgt_grouped_moe")

    score = torch.compile(
        forward, fullgraph=True, dynamic=False, mode="reduce-overhead"
    )
    sequences = ["ACGT" * 2048]  # Replace with a batch of complete ACGTN windows.
    ids = torch.from_numpy(
        encode_dna_batch(compact_dna_unigram_tokenizer(), sequences)
    ).to(device)
    with torch.inference_mode():
        logits = score(ids).float().cpu().numpy()
    print("A/C/G/T logits:", logits.shape)  # [batch, 8192, 4]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, help="Reuse an existing HF download")
    main(parser.parse_args().model_dir)
