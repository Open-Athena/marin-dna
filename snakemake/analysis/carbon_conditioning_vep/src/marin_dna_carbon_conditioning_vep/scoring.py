"""Prompt-aware Carbon full-sequence likelihood scoring.

The token likelihood calculation is copied from Carbon's pinned
``evaluation/vep_eval.py`` at commit
``10bbc4b35f6e26d2a8767342576ff65108028bf5`` and adapted to accept a frozen
metadata prefix.
"""

from __future__ import annotations

import multiprocessing as mp
import resource
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna_carbon_conditioning_vep.dna import reverse_complement
from marin_dna_carbon_conditioning_vep.prompts import (
    assert_allele_token_count_parity,
    assert_prefix_outside_dna_mode,
    render_prompt,
)

SCORE_COLUMNS = (
    "ll_ref_fwd",
    "ll_alt_fwd",
    "ll_ref_rc",
    "ll_alt_rc",
    "llr_fwd",
    "llr_rc",
    "llr",
    "score",
)


@dataclass(frozen=True)
class ShardRuntime:
    """Execution metadata returned by one GPU scoring process."""

    shard_id: int
    device: str
    rows: int
    elapsed_seconds: float
    peak_gpu_memory_bytes: int
    peak_rss_bytes: int


def masked_mean_causal_log_likelihood(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Match Carbon's pinned masked mean next-token log-likelihood calculation."""
    assert input_ids.ndim == 2 and attention_mask.shape == input_ids.shape
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].to(dtype=torch.float32)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    token_log_probabilities = log_probabilities.gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)
    denominator = mask.sum(dim=1).clamp(min=1)
    return (token_log_probabilities * mask).sum(dim=1) / denominator


def derive_score_atoms(
    ll_ref_fwd: np.ndarray,
    ll_alt_fwd: np.ndarray,
    ll_ref_rc: np.ndarray,
    ll_alt_rc: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-strand LLR, strand average, and deleteriousness orientation."""
    arrays = [
        np.asarray(values, dtype=float)
        for values in (ll_ref_fwd, ll_alt_fwd, ll_ref_rc, ll_alt_rc)
    ]
    lengths = {len(values) for values in arrays}
    assert len(lengths) == 1, f"likelihood arrays have different lengths: {lengths}"
    llr_fwd = arrays[1] - arrays[0]
    llr_rc = arrays[3] - arrays[2]
    llr = (llr_fwd + llr_rc) / 2.0
    return {
        "ll_ref_fwd": arrays[0],
        "ll_alt_fwd": arrays[1],
        "ll_ref_rc": arrays[2],
        "ll_alt_rc": arrays[3],
        "llr_fwd": llr_fwd,
        "llr_rc": llr_rc,
        "llr": llr,
        "score": -llr,
    }


def _peak_rss_bytes() -> int:
    """Return Linux ``ru_maxrss`` in bytes."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def load_carbon_model_and_tokenizer(
    model_repo: str,
    model_revision: str,
    device: torch.device,
    dtype_name: str,
) -> tuple[Any, Any]:
    assert dtype_name == "bfloat16", f"issue #486 fixes bf16, got {dtype_name!r}"
    tokenizer = AutoTokenizer.from_pretrained(
        model_repo,
        revision=model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        revision=model_revision,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    return model, tokenizer


def _score_prompts(
    prompts: list[str],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> np.ndarray:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        values = masked_mean_causal_log_likelihood(model, input_ids, attention_mask)
    return values.detach().to(device="cpu", dtype=torch.float32).numpy()


def _score_shard(
    shard_id: int,
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    model_repo: str,
    model_revision: str,
    dtype_name: str,
    batch_size: int,
    kmer_size: int,
) -> tuple[list[dict[str, Any]], ShardRuntime]:
    start_time = time.monotonic()
    torch.cuda.set_device(shard_id)
    torch.cuda.reset_peak_memory_stats(shard_id)
    device = torch.device("cuda", shard_id)
    model, tokenizer = load_carbon_model_and_tokenizer(
        model_repo, model_revision, device, dtype_name
    )
    assert_prefix_outside_dna_mode(tokenizer, prefix)
    output: list[dict[str, Any]] = []
    for offset in tqdm(
        range(0, len(rows), batch_size),
        desc=f"gpu{shard_id}",
        unit="batch",
    ):
        batch = rows[offset : offset + batch_size]
        prompts: list[str] = []
        for row in batch:
            ref = str(row["ref_sequence"])
            alt = str(row["alt_sequence"])
            ref_rc = reverse_complement(ref)
            alt_rc = reverse_complement(alt)
            assert_allele_token_count_parity(tokenizer, prefix, ref, alt, kmer_size)
            assert_allele_token_count_parity(
                tokenizer, prefix, ref_rc, alt_rc, kmer_size
            )
            prompts.extend(
                [
                    render_prompt(prefix, ref, kmer_size),
                    render_prompt(prefix, alt, kmer_size),
                    render_prompt(prefix, ref_rc, kmer_size),
                    render_prompt(prefix, alt_rc, kmer_size),
                ]
            )
        values = _score_prompts(
            prompts,
            model=model,
            tokenizer=tokenizer,
            device=device,
        ).reshape(len(batch), 4)
        atoms = derive_score_atoms(
            values[:, 0], values[:, 1], values[:, 2], values[:, 3]
        )
        for row_index, row in enumerate(batch):
            result = {"row_index": int(row["row_index"])}
            result.update(
                {name: float(atoms[name][row_index]) for name in SCORE_COLUMNS}
            )
            output.append(result)

    runtime = ShardRuntime(
        shard_id=shard_id,
        device=torch.cuda.get_device_name(shard_id),
        rows=len(rows),
        elapsed_seconds=time.monotonic() - start_time,
        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(shard_id)),
        peak_rss_bytes=_peak_rss_bytes(),
    )
    return output, runtime


def _score_shard_entrypoint(
    arguments: tuple[int, list[dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], ShardRuntime]:
    shard_id, rows, kwargs = arguments
    return _score_shard(shard_id, rows, **kwargs)


def score_condition_dataframe(
    windows: pd.DataFrame,
    *,
    condition: str,
    prompt_grammar: str,
    prefix: str,
    model_repo: str,
    model_revision: str,
    dtype_name: str,
    batch_size: int,
    kmer_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one prompt condition across all visible GPUs."""
    required = {"variant_id", "ref_sequence", "alt_sequence"}
    missing = sorted(required - set(windows.columns))
    assert not missing, f"window parquet missing columns: {missing}"
    assert not windows["variant_id"].duplicated().any(), "variant_id must be unique"
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("Carbon scoring requires at least one CUDA GPU")

    indexed = windows.reset_index(drop=True).copy()
    indexed.insert(0, "row_index", np.arange(len(indexed), dtype=np.int64))
    shard_rows = [
        indexed.iloc[shard_id::n_gpus][
            ["row_index", "ref_sequence", "alt_sequence"]
        ].to_dict("records")
        for shard_id in range(n_gpus)
    ]
    worker_kwargs = {
        "prefix": prefix,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "dtype_name": dtype_name,
        "batch_size": batch_size,
        "kmer_size": kmer_size,
    }
    start_time = time.monotonic()
    work = [
        (shard_id, rows, worker_kwargs)
        for shard_id, rows in enumerate(shard_rows)
        if rows
    ]
    if len(work) == 1:
        shard_results = [_score_shard_entrypoint(work[0])]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(work)) as pool:
            shard_results = list(pool.map(_score_shard_entrypoint, work))

    scores = pd.DataFrame(
        [row for shard, _runtime in shard_results for row in shard]
    ).sort_values("row_index")
    assert scores["row_index"].tolist() == list(range(len(indexed))), (
        "GPU shard results are missing, duplicated, or reordered"
    )
    metadata = indexed.drop(columns=["row_index", "ref_sequence", "alt_sequence"])
    result = pd.concat(
        [
            metadata.reset_index(drop=True),
            scores.drop(columns="row_index").reset_index(drop=True),
        ],
        axis=1,
    )
    result["condition"] = condition
    result["prompt_grammar"] = prompt_grammar
    result["prompt_prefix"] = prefix
    result["model"] = model_repo
    result["model_revision"] = model_revision
    assert len(result) == len(windows)
    assert np.isfinite(result[list(SCORE_COLUMNS)].to_numpy(dtype=float)).all()

    runtimes = [runtime for _rows, runtime in shard_results]
    runtime_summary: dict[str, Any] = {
        "condition": condition,
        "prompt_grammar": prompt_grammar,
        "prompt_prefix": prefix,
        "model": model_repo,
        "model_revision": model_revision,
        "dtype": dtype_name,
        "rows": len(result),
        "gpu_count": len(runtimes),
        "devices": sorted({runtime.device for runtime in runtimes}),
        "elapsed_seconds": time.monotonic() - start_time,
        "peak_gpu_memory_bytes": max(
            runtime.peak_gpu_memory_bytes for runtime in runtimes
        ),
        "peak_rss_bytes": max(runtime.peak_rss_bytes for runtime in runtimes),
        "shards": [asdict(runtime) for runtime in runtimes],
    }
    return result, runtime_summary
