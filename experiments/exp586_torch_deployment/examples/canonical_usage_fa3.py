"""Canonical usage for torch-fa3: native Torch experts with standalone FA3."""

import torch

from exp586_deployment import MarinDNAMoE
from exp586_deployment.tokenizer import (
    dna_nucleotide_tokenizer,
    encode_dna_batch,
)

model = MarinDNAMoE.from_pretrained(
    "plantcad/marin-dna-exp586-1.5B-60M-216B",
    revision="4130f71edcd117a7375af1ac78723d9a9173533f",
    device="cuda",
    use_vllm=False,  # Standalone attention and native Torch experts.
    flash_attention_version=3,
)
forward = torch.compile(model, fullgraph=True)

tokenizer = dna_nucleotide_tokenizer()
input_ids = torch.from_numpy(encode_dna_batch(tokenizer, ["ACGT" * 2048])).to("cuda")

with torch.inference_mode():
    logits = forward(input_ids)  # [1, 8192, 4], columns A/C/G/T.
