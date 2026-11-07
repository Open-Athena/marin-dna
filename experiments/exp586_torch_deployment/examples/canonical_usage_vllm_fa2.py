"""Canonical inference example; use the documented pinned environment.

Ordinary examples: env -u TORCHINDUCTOR_FREEZING python <example.py>
Fast example: TORCHINDUCTOR_FREEZING=1 python canonical_usage_fast.py
"""

import os

os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    OMP_NUM_THREADS="1",
    TORCHINDUCTOR_COMPILE_THREADS="1",
)

import random

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
    dtype=torch.bfloat16,
    input_vocab="ACGTN",
    output_vocab="ACGT",
    attention_provider="vllm",
    flash_attention_version=2,
    moe_backend="vllm",
    sconv_backend="direct",
    cache_rope=True,
)
forward = torch.compile(model, fullgraph=True)

rng = random.Random(0)
sequences = ["".join(rng.choices("ACGT", k=8192)) for _ in range(64)]

# To use the original benchmark inputs instead:
# from pathlib import Path
# from exp586_deployment.fixtures import load_fixture
# sequences = load_fixture(Path("benchmark-fixture-512.json"))[:64]

tokenizer = dna_nucleotide_tokenizer()
input_ids = torch.from_numpy(encode_dna_batch(tokenizer, sequences)).to("cuda")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

with torch.inference_mode():
    # Compile and warm up before timing.
    for _ in range(8):
        forward(input_ids)
        torch.cuda.synchronize()

    start.record()
    logits = forward(input_ids)  # [64, 8192, 4], columns A/C/G/T.
    end.record()
    end.synchronize()

elapsed_ms = start.elapsed_time(end)
print(f"{elapsed_ms:.1f} ms; {input_ids.numel() * 1000 / elapsed_ms:,.0f} tokens/s")
