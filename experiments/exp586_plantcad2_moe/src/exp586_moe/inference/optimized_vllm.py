"""Simple single-H100 interface for the optimized vLLM logits path."""

import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from exp586_moe.inference.benchmark_vllm import _clear_captures, _install_capture
from exp586_moe.inference.contract import validate_sequences
from exp586_moe.inference.tokenizer import (
    encode_dna_batch_lists,
    reduced_dna_unigram_tokenizer,
)


def _write_captures(worker, output: str, real_rows: int) -> tuple[int, ...]:
    """Write captured worker outputs without serializing them through RPC."""
    captures = worker.model_runner.model._exp586_captures
    values = np.concatenate(captures, axis=0)[:real_rows]
    np.save(output, values)
    captures.clear()
    return values.shape


class OptimizedVllmDnaLogits:
    """Score full 8,192-base windows with the fastest validated vLLM path."""

    output_bases = "ACGT"

    def __init__(
        self,
        model_dir: str | Path,
        *,
        batch_size: int = 768,
        scheduler_token_budget: int = 196_608,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if scheduler_token_budget < 8192 or scheduler_token_budget % 8192:
            raise ValueError("scheduler_token_budget must be a multiple of 8,192")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        import torch
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA device to this scorer")
        if "H100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This configuration was optimized and validated on H100")

        directory = Path(model_dir)
        source_tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer = reduced_dna_unigram_tokenizer(source_tokenizer)
        base_ids = [
            self.tokenizer.token_to_id(base.lower()) for base in self.output_bases
        ]
        if any(token_id is None for token_id in base_ids):
            raise ValueError("Tokenizer lacks isolated A/C/G/T tokens")

        self.batch_size = batch_size
        self.sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
        self.llm = LLM(
            model=str(directory),
            dtype="bfloat16",
            tensor_parallel_size=1,
            enforce_eager=False,
            max_model_len=8193,
            max_num_batched_tokens=scheduler_token_budget,
            max_num_seqs=batch_size,
            gpu_memory_utilization=0.2,
            kv_cache_memory_bytes=8_267_812_045,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            disable_log_stats=True,
            moe_backend="triton",
            attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 3},
        )
        installed = self.llm.llm_engine.collective_rpc(
            _install_capture,
            args=(
                [int(token_id) for token_id in base_ids],
                "acgt_head",
                "float32",
                False,
                True,
                True,
                None,
            ),
        )
        if installed != [True]:
            raise RuntimeError(f"Capture hook was not installed: {installed}")

    def _prompts(self, sequences: list[str]) -> list[dict[str, list[int]]]:
        ids = encode_dna_batch_lists(self.tokenizer, sequences)
        return [{"prompt_token_ids": sequence_ids} for sequence_ids in ids]

    def warmup(self, sequence: str) -> None:
        """Compile and warm the engine before scoring."""
        validate_sequences([sequence])
        self.llm.generate(self._prompts([sequence]), self.sampling, use_tqdm=False)
        self.llm.llm_engine.collective_rpc(_clear_captures)

    def score_to_directory(
        self, sequences: Sequence[str], output_dir: str | Path
    ) -> Iterator[tuple[Path, tuple[int, ...]]]:
        """Write FP32 ``[batch, 8192, 4]`` arrays in A/C/G/T order."""
        validate_sequences(sequences)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for index, start in enumerate(range(0, len(sequences), self.batch_size)):
            batch = list(sequences[start : start + self.batch_size])
            self.llm.generate(self._prompts(batch), self.sampling, use_tqdm=False)
            output = destination / f"logits-{index:05d}.npy"
            shapes = self.llm.llm_engine.collective_rpc(
                _write_captures,
                args=(str(output), len(batch)),
            )
            if len(shapes) != 1:
                raise RuntimeError(f"Expected one vLLM worker, got {shapes}")
            yield output, tuple(int(value) for value in shapes[0])
