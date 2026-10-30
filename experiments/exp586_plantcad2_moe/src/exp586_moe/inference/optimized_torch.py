"""Simple single-H100 interface for the optimized PyTorch logits path."""

import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

# Freeze inference parameters before Torch/Inductor is imported.  This enables
# the constant-weight graph used by the final validated benchmark.
os.environ.setdefault("TORCHINDUCTOR_FREEZING", "1")

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from exp586_moe.inference.contract import token_contract, validate_sequences
from exp586_moe.inference.optimized_common import padded_batches
from exp586_moe.inference.tokenizer import (
    compact_dna_unigram_tokenizer,
    encode_dna_batch,
)
from exp586_moe.inference.torch_backend import EagerGrugMoE


class OptimizedTorchDnaLogits:
    """Score full 8,192-base windows with the fastest validated Torch path."""

    output_bases = "ACGT"

    def __init__(self, model_dir: str | Path, *, batch_size: int = 240) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA device to this scorer")
        device = torch.device("cuda:0")
        if "H100" not in torch.cuda.get_device_name(device):
            raise RuntimeError("This configuration was optimized and validated on H100")

        directory = Path(model_dir)
        source_tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
        self.tokenizer = compact_dna_unigram_tokenizer()
        contract = token_contract(source_tokenizer.backend_tokenizer)
        self.base_ids = [contract.output_ids[base] for base in self.output_bases]
        self.batch_size = batch_size
        self.device = device
        self.model = EagerGrugMoE(
            directory,
            device,
            attention_backend="vllm_fa3",
            global_attention_backend="flash",
            specialized_topk=True,
            sconv_backend="direct",
            compact_input_ids=[contract.input_ids[base] for base in "ACGTN"],
            compact_output_ids=self.base_ids,
            precomputed_rope=True,
            combined_shared_expert_projections=True,
        )

        def forward(token_ids: torch.Tensor) -> torch.Tensor:
            return self.model(token_ids, self.base_ids, variant="acgt_vllm_moe")

        self.forward = torch.compile(
            forward,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )

    def _tokenize(self, sequences: list[str]) -> torch.Tensor:
        ids = encode_dna_batch(self.tokenizer, sequences)
        return torch.from_numpy(ids)

    def _forward(self, token_ids: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            logits = self.forward(token_ids.to(self.device))
        return logits.float().cpu().numpy()

    def warmup(self, sequence: str) -> None:
        """Compile and warm the fixed batch shape before timed scoring."""
        validate_sequences([sequence])
        batch, _ = next(padded_batches([sequence], self.batch_size))
        token_ids = self._tokenize(batch)
        self._forward(token_ids)

    def score_batches(self, sequences: Sequence[str]) -> Iterator[np.ndarray]:
        """Yield FP32 arrays shaped ``[batch, 8192, 4]`` in A/C/G/T order."""
        batches = iter(padded_batches(sequences, self.batch_size))
        try:
            first_batch = next(batches)
        except StopIteration:
            return

        with ThreadPoolExecutor(max_workers=1) as pool:
            batch, real_rows = first_batch
            pending = pool.submit(self._tokenize, batch)
            for next_batch in chain(batches, (None,)):
                token_ids = pending.result()
                if next_batch is not None:
                    batch, next_real_rows = next_batch
                    pending = pool.submit(self._tokenize, batch)
                logits = self._forward(token_ids)
                yield logits[:real_rows]
                if next_batch is not None:
                    real_rows = next_real_rows
