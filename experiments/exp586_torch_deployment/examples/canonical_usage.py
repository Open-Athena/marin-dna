"""Canonical usage: fetch the HF model and run the ordinary configuration.

Keep this example as the named reference for future deployment discussions.
Change its contract only when a deliberate revision has been agreed.
"""

import torch

from exp586_deployment import MarinDNAMoE
from exp586_deployment.tokenizer import (
    compact_dna_unigram_tokenizer,
    encode_dna_batch,
)

model = MarinDNAMoE.from_pretrained(
    "plantcad/marin-dna-exp586-1.5B-60M-216B",
    revision="4130f71edcd117a7375af1ac78723d9a9173533f",
    device="cuda",
    use_vllm=False,  # True requires vLLM and uses fused experts.
    flash_attention_version=2,  # 2 or 3.
)
forward = torch.compile(model, fullgraph=True)

tokenizer = compact_dna_unigram_tokenizer()
input_ids = torch.from_numpy(encode_dna_batch(tokenizer, ["ACGT" * 2048])).to("cuda")

with torch.inference_mode():
    logits = forward(input_ids)  # [1, 8192, 4], columns A/C/G/T.
