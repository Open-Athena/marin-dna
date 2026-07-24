#!/usr/bin/env python3
"""Score one frozen issue #402 RAG benchmark with an exported HF checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.pipelines.rag_glm.offline_eval import (
    RAG_BENCHMARK_DATASETS,
    aggregate_rag_variant_scores,
    compute_rag_benchmark_metrics,
    load_rag_eval_split,
    score_rag_rows_hf,
    write_rag_evaluation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=sorted(RAG_BENCHMARK_DATASETS), required=True
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--model", required=True, help="Local path or Hugging Face model ID"
    )
    parser.add_argument(
        "--model-revision", help="Required full SHA for remote model IDs"
    )
    parser.add_argument("--tokenizer", help="Tokenizer path/ID; defaults to --model")
    parser.add_argument(
        "--dataset-repo", help="Override the frozen benchmark repository"
    )
    parser.add_argument(
        "--dataset-revision", help="Override with a full dataset commit SHA"
    )
    parser.add_argument(
        "--code-revision", required=True, help="Full marin-dna commit SHA"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_repo, default_revision = RAG_BENCHMARK_DATASETS[args.benchmark]
    assert args.dataset_repo is None or args.dataset_revision is not None, (
        "--dataset-repo requires --dataset-revision"
    )
    dataset_repo = args.dataset_repo or default_repo
    dataset_revision = args.dataset_revision or default_revision
    if not Path(args.model).exists():
        assert args.model_revision and len(args.model_revision) == 40, (
            "remote models require --model-revision with a full commit SHA"
        )
    assert len(args.code_revision) == 40
    assert len(dataset_revision) == 40

    pretrained_kwargs: dict[str, object] = {"trust_remote_code": True}
    if args.model_revision is not None:
        pretrained_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, **pretrained_kwargs
    )
    model_kwargs = dict(pretrained_kwargs)
    if torch.device(args.device).type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(args.device)

    rows = load_rag_eval_split(
        args.benchmark,
        args.split,
        repo=dataset_repo,
        revision=dataset_revision,
    )
    documents = score_rag_rows_hf(
        model,
        tokenizer,
        rows,
        batch_size=args.batch_size,
        device=args.device,
    )
    variants = aggregate_rag_variant_scores(documents, args.benchmark)
    metrics = compute_rag_benchmark_metrics(
        variants,
        args.benchmark,
        n_bootstrap=args.n_bootstrap,
    )
    write_rag_evaluation_outputs(
        document_scores=documents,
        variant_scores=variants,
        metrics=metrics,
        output_dir=args.output_dir,
        benchmark=args.benchmark,
        split=args.split,
        model=args.model,
        model_revision=args.model_revision,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        code_revision=args.code_revision,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
