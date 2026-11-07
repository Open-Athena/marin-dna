"""Canonical usage for fast-fa3: custom router, vLLM fused experts, FA3.

Run in a fresh Python process so freezing is set before Torch is imported.
The project-specific Triton router is JIT-compiled for the installed GPU stack.
"""

import os

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

import torch

from exp586_deployment import FastMarinDNAMoE
from exp586_deployment.tokenizer import (
    compact_dna_unigram_tokenizer,
    encode_dna_batch,
)

model = FastMarinDNAMoE.from_pretrained(
    "plantcad/marin-dna-exp586-1.5B-60M-216B",
    revision="4130f71edcd117a7375af1ac78723d9a9173533f",
    device="cuda",
    use_vllm=True,
    flash_attention_version=3,
)
forward = torch.compile(model, fullgraph=True, dynamic=False, mode="reduce-overhead")

tokenizer = compact_dna_unigram_tokenizer()
input_ids = torch.from_numpy(encode_dna_batch(tokenizer, ["ACGT" * 2048])).to("cuda")

with torch.inference_mode():
    logits = forward(input_ids)  # [1, 8192, 4], columns A/C/G/T.
