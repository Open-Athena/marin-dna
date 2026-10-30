"""Simple single-H100 interface for the optimized native-JAX logits path."""

import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

# The batch-352 frontier requires the same 95% device pool used in validation.
# XLA reads this setting when the backend is initialized.
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.95")

import jax
import numpy as np

from exp586_moe.inference.contract import validate_sequences
from exp586_moe.inference.jax_backend import (
    compile_forward,
    configure_fa4_forward,
    encode,
    load_model,
)
from exp586_moe.inference.optimized_common import padded_batches
from exp586_moe.inference.tokenizer import compact_dna_unigram_tokenizer


class OptimizedJaxDnaLogits:
    """Score full 8,192-base windows with the fastest validated JAX path."""

    output_bases = "ACGT"

    def __init__(self, *, batch_size: int = 352) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if jax.device_count() != 1 or "H100" not in jax.devices()[0].device_kind:
            raise RuntimeError("Expose exactly one H100 to this scorer")
        configure_fa4_forward(tile_m=64, tile_n=128, num_threads=128)
        self.tokenizer = compact_dna_unigram_tokenizer()
        model, mesh, base_ids = load_model(
            compact_vocab=True,
            moe_implementation="sonic",
            attention_implementation="gpu_fa4_cute",
        )
        self.forward = compile_forward(
            model,
            mesh,
            base_ids,
            variant="acgt_head",
            output_dtype="float32",
        )
        self.batch_size = batch_size

    def _tokenize(self, sequences: list[str]) -> np.ndarray:
        return encode(self.tokenizer, sequences)

    def _forward(self, token_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.forward(jax.device_put(token_ids)))

    def warmup(self, sequence: str) -> None:
        """Compile and warm the fixed batch shape before timed scoring."""
        validate_sequences([sequence])
        batch, _ = next(padded_batches([sequence], self.batch_size))
        self._forward(self._tokenize(batch))

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
