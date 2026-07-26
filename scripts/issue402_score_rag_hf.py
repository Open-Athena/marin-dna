#!/usr/bin/env python3
"""Score one frozen issue #402 RAG benchmark with an exported HF checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from marin_dna.pipelines.rag_glm.offline_eval import (
    RAG_BENCHMARK_DATASETS,
    aggregate_rag_variant_scores,
    compute_rag_benchmark_metrics,
    load_rag_eval_split,
    load_rag_model_config_hf,
    load_rag_tokenizer_hf,
    encode_rag_batch,
    nucleotide_token_ids,
    run_rag_mendelian_probe,
    score_rag_rows_hf,
    select_paired_rag_rows,
    write_rag_evaluation_outputs,
    write_rag_probe_outputs,
)
from marin_dna.pipelines.rag_glm.hf_scoring import (
    score_rag_completions_hf,
    score_rag_completions_naive_hf,
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
        "--model-source",
        help="Immutable source URI/ID recorded in outputs; defaults to --model",
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
        "--max-rows",
        type=int,
        help="Score only the first N rows, asserting complete fwd/rc variant pairs",
    )
    parser.add_argument(
        "--return-embeddings",
        action="store_true",
        help="Pool the 255-token human segment for each allele and strand",
    )
    parser.add_argument(
        "--run-probe",
        action="store_true",
        help="Run the frozen Mendelian chromosome-held-out linear probe",
    )
    parser.add_argument("--probe-n-jobs", type=int, default=4)
    parser.add_argument(
        "--verify-naive-only",
        action="store_true",
        help="Print metadata-only cached/naive score pairs and exit before metrics",
    )
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
    model_source = args.model_source or args.model
    assert model_source
    assert not args.run_probe or args.benchmark == "mendelian_traits", (
        "--run-probe is frozen for the Mendelian benchmark"
    )
    assert not args.run_probe or args.return_embeddings, (
        "--run-probe requires --return-embeddings"
    )
    assert args.probe_n_jobs > 0
    assert args.max_rows is None or args.max_rows > 0
    assert not args.verify_naive_only or args.max_rows is not None
    assert not args.verify_naive_only or args.max_rows <= 16

    pretrained_kwargs: dict[str, object] = {"trust_remote_code": True}
    if args.model_revision is not None:
        pretrained_kwargs["revision"] = args.model_revision
    tokenizer = load_rag_tokenizer_hf(
        args.tokenizer or args.model,
        revision=args.model_revision,
    )
    model_config = load_rag_model_config_hf(
        args.model,
        revision=args.model_revision,
    )
    model_kwargs = dict(pretrained_kwargs)
    model_kwargs["config"] = model_config
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
    rows = select_paired_rag_rows(rows, args.max_rows)
    if args.verify_naive_only:
        prefix, ref, alt = encode_rag_batch(tokenizer, rows)
        prefix = prefix.to(args.device)
        ref = ref.to(args.device)
        alt = alt.to(args.device)
        nucleotides = nucleotide_token_ids(tokenizer).to(args.device)
        with torch.inference_mode():
            cached = (
                score_rag_completions_hf(
                    model,
                    prefix,
                    ref,
                    alt,
                    nucleotide_token_ids=nucleotides,
                )
                .float()
                .cpu()
            )
            naive = (
                score_rag_completions_naive_hf(
                    model,
                    prefix,
                    ref,
                    alt,
                    nucleotide_token_ids=nucleotides,
                )
                .float()
                .cpu()
            )
        assert torch.isfinite(cached).all()
        assert torch.isfinite(naive).all()
        for index, row in enumerate(rows.iter_rows(named=True)):
            print(
                "RAG_HF_PARITY_ROW "
                + json.dumps(
                    {
                        "document_id": row["document_id"],
                        "cached_ref_loglikelihood": float(cached[index, 0]),
                        "cached_alt_loglikelihood": float(cached[index, 1]),
                        "cached_llr": float(cached[index, 2]),
                        "naive_ref_loglikelihood": float(naive[index, 0]),
                        "naive_alt_loglikelihood": float(naive[index, 1]),
                        "naive_llr": float(naive[index, 2]),
                    },
                    sort_keys=True,
                )
            )
        print(f"RAG_HF_PARITY_MAX_ABS {float((cached - naive).abs().max()):.9g}")
        return
    documents = score_rag_rows_hf(
        model,
        tokenizer,
        rows,
        batch_size=args.batch_size,
        device=args.device,
        return_embeddings=args.return_embeddings,
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
        model_source=model_source,
        model_revision=args.model_revision,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        code_revision=args.code_revision,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    if args.run_probe:
        predictions, probe_metrics, classifiers = run_rag_mendelian_probe(
            variants,
            n_jobs=args.probe_n_jobs,
            n_bootstrap=args.n_bootstrap,
        )
        write_rag_probe_outputs(
            predictions=predictions,
            metrics=probe_metrics,
            classifiers=classifiers,
            output_dir=args.output_dir,
            model=args.model,
            model_source=model_source,
            model_revision=args.model_revision,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            code_revision=args.code_revision,
        )


if __name__ == "__main__":
    main()
