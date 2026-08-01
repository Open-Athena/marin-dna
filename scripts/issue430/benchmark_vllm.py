"""Benchmark exact MarinDNA VEP prompt scoring with vLLM for issue #430."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from marin_dna.pipelines.evals.inference_benchmark import (  # noqa: E402
    PreparedHarnessLlr,
    VARIANT_KEY_COLUMNS,
    aggregate_harness_llr,
    prepare_harness_llr,
)


MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_ID = "marin-dna/evals_mendelian_traits_harness_255"
DATASET_REVISION = "7b92f047f9a36f90e9ac47886afa2a99264ee35c"


def _logsumexp(values: np.ndarray) -> float:
    assert values.ndim == 1 and len(values) > 0
    maximum = float(values.max())
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _score_prompt_batch(
    llm: Any,
    sampling_params: Any,
    prompts: np.ndarray,
    *,
    var_pos: int,
    nucleotide_token_ids: np.ndarray,
) -> np.ndarray:
    outputs = llm.generate(
        [prompt.tolist() for prompt in prompts],
        sampling_params,
        use_tqdm=False,
    )
    assert len(outputs) == len(prompts)
    nucleotide_ids = nucleotide_token_ids.tolist()
    nucleotide_id_set = set(nucleotide_ids)
    prompt_scores = np.empty(len(prompts), dtype=np.float64)

    for output_index, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
        assert output.prompt_token_ids == prompt.tolist()
        prompt_logprobs = output.prompt_logprobs
        assert prompt_logprobs is not None
        assert len(prompt_logprobs) == len(prompt)
        score = 0.0
        for position in range(var_pos, len(prompt)):
            token_id = int(prompt[position])
            assert token_id in nucleotide_id_set
            logprobs = prompt_logprobs[position]
            assert logprobs is not None
            assert nucleotide_id_set.issubset(logprobs)
            nucleotide_logprobs = np.array(
                [logprobs[nuc_id].logprob for nuc_id in nucleotide_ids],
                dtype=np.float64,
            )
            score += float(logprobs[token_id].logprob) - _logsumexp(nucleotide_logprobs)
        prompt_scores[output_index] = score

    assert np.isfinite(prompt_scores).all()
    return prompt_scores


def _score_rows(
    llm: Any,
    sampling_params: Any,
    prompts: np.ndarray,
    *,
    request_batch_size: int,
    var_pos: int,
    nucleotide_token_ids: np.ndarray,
) -> np.ndarray:
    assert prompts.ndim == 2 and len(prompts) % 2 == 0
    prompt_scores = np.empty(len(prompts), dtype=np.float64)
    for start in range(0, len(prompts), request_batch_size):
        stop = min(start + request_batch_size, len(prompts))
        prompt_scores[start:stop] = _score_prompt_batch(
            llm,
            sampling_params,
            prompts[start:stop],
            var_pos=var_pos,
            nucleotide_token_ids=nucleotide_token_ids,
        )
    row_llr = prompt_scores[0::2] - prompt_scores[1::2]
    assert np.isfinite(row_llr).all()
    return row_llr


def _select_variants(
    prepared: PreparedHarnessLlr,
    max_variants: int | None,
) -> PreparedHarnessLlr:
    if max_variants is None:
        return prepared
    assert max_variants > 0
    group_id = prepared.metadata.groupby(
        VARIANT_KEY_COLUMNS, sort=False, dropna=False
    ).ngroup()
    indices = np.flatnonzero(group_id.to_numpy() < max_variants)
    assert len(indices) == 2 * max_variants
    return PreparedHarnessLlr(
        metadata=prepared.metadata.iloc[indices].reset_index(drop=True),
        input_ids=prepared.input_ids[indices],
        alt_token_id=prepared.alt_token_id[indices],
        var_pos=prepared.var_pos,
        nuc_token_ids=prepared.nuc_token_ids,
    )


def _build_prompts(prepared: PreparedHarnessLlr) -> np.ndarray:
    reference = prepared.input_ids.numpy().astype(np.int32, copy=True)
    alternate = reference.copy()
    alternate[:, prepared.var_pos] = prepared.alt_token_id.numpy()
    prompts = np.empty((2 * len(reference), reference.shape[1]), dtype=np.int32)
    prompts[0::2] = reference
    prompts[1::2] = alternate
    assert prompts.shape == (2 * len(prepared.metadata), 256)
    return prompts


def _compare_reference(
    candidate: pd.DataFrame,
    reference_path: Path,
) -> dict[str, float]:
    reference = pd.read_parquet(reference_path)
    merged = candidate.merge(
        reference[[*VARIANT_KEY_COLUMNS, "minus_llr_avg"]],
        on=VARIANT_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    assert len(merged) == len(candidate)
    assert merged["minus_llr_avg_reference"].notna().all()
    delta = (
        merged["minus_llr_avg_candidate"] - merged["minus_llr_avg_reference"]
    ).to_numpy()
    return {
        "max_abs_minus_llr_avg_delta": float(np.abs(delta).max()),
        "mean_abs_minus_llr_avg_delta": float(np.abs(delta).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path.home() / "ckpt"))
    parser.add_argument("--subset", default="missense_variant")
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--request-batch-size", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--price-per-hour", type=float, default=3.29)
    parser.add_argument("--reference-scores", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    assert args.request_batch_size > 0
    assert args.repetitions > 0

    # vLLM requires at least one generated token even when only prompt
    # log-probabilities are consumed. The scored prompt remains exactly 256
    # tokens; permit one unscored decode slot beyond the checkpoint's declared
    # context length.
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    from vllm import LLM, SamplingParams

    preprocessing_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    harness = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
    ).to_pandas()
    prepared = prepare_harness_llr(harness, tokenizer, subset=args.subset)
    prepared = _select_variants(prepared, args.max_variants)
    prompts = _build_prompts(prepared)
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    engine_start = time.perf_counter()
    llm = LLM(
        model=args.checkpoint,
        tokenizer=args.checkpoint,
        dtype="bfloat16",
        max_model_len=257,
        max_num_seqs=args.request_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
        logprobs_mode="raw_logprobs",
    )
    engine_load_seconds = time.perf_counter() - engine_start
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=-1,
        detokenize=False,
    )
    nucleotide_token_ids = prepared.nuc_token_ids.numpy()

    warmup_start = time.perf_counter()
    _score_rows(
        llm,
        sampling_params,
        prompts[:2],
        request_batch_size=2,
        var_pos=prepared.var_pos,
        nucleotide_token_ids=nucleotide_token_ids,
    )
    warmup_seconds = time.perf_counter() - warmup_start

    repeat_seconds: list[float] = []
    repeat_llr: list[np.ndarray] = []
    for _ in range(args.repetitions):
        reset_prefix_cache = getattr(llm, "reset_prefix_cache", None)
        if reset_prefix_cache is not None:
            reset_prefix_cache()
        start = time.perf_counter()
        row_llr = _score_rows(
            llm,
            sampling_params,
            prompts,
            request_batch_size=args.request_batch_size,
            var_pos=prepared.var_pos,
            nucleotide_token_ids=nucleotide_token_ids,
        )
        repeat_seconds.append(time.perf_counter() - start)
        repeat_llr.append(row_llr)

    for candidate in repeat_llr[1:]:
        np.testing.assert_allclose(candidate, repeat_llr[0], atol=1e-6, rtol=0)
    row_indices = np.arange(len(prepared.metadata), dtype=np.int64)
    scores = aggregate_harness_llr(prepared, row_indices, repeat_llr[0])
    auprc = float(
        average_precision_score(scores["target"].astype(int), scores["minus_llr_avg"])
    )
    median_seconds = float(np.median(repeat_seconds))
    variants_per_second = len(scores) / median_seconds
    seconds_per_million = 1_000_000 / variants_per_second

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out_dir / "scores.parquet", index=False)
    row_scores = prepared.metadata.copy()
    row_scores["llr"] = repeat_llr[0]
    row_scores.to_parquet(args.out_dir / "row_scores.parquet", index=False)
    summary = {
        "model": "marin-dna/marin-dna-exp135-m5.1",
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "subset": args.subset,
        "max_variants": args.max_variants,
        "n_variants": len(scores),
        "n_strand_rows": len(prepared.metadata),
        "n_prompts": len(prompts),
        "scored_prompt_tokens": 256,
        "engine_max_model_len": 257,
        "request_batch_size": args.request_batch_size,
        "enforce_eager": args.enforce_eager,
        "preprocessing_seconds": preprocessing_seconds,
        "engine_load_seconds": engine_load_seconds,
        "warmup_seconds": warmup_seconds,
        "repeat_seconds": repeat_seconds,
        "median_seconds": median_seconds,
        "variants_per_second": variants_per_second,
        "variants_per_hour": variants_per_second * 3600,
        "seconds_per_million": seconds_per_million,
        "price_per_hour": args.price_per_hour,
        "dollars_per_million": seconds_per_million / 3600 * args.price_per_hour,
        "auprc": auprc,
        "vllm": importlib.metadata.version("vllm"),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
    }
    if args.reference_scores is not None:
        summary["reference_parity"] = _compare_reference(scores, args.reference_scores)
    (args.out_dir / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
