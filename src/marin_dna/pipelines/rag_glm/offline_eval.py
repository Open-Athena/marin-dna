"""Offline Hugging Face evaluation for issue #402's RAG VEP harnesses.

The harness parquets already contain exact HAL-projected ortholog windows, so
this module needs no genome or alignment files. It scores every materialized
strand document, averages the raw LLR across strands, and only then applies the
benchmark's deleteriousness transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from torch import Tensor
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from marin_dna.pipelines.evals.metrics import compute_auprc_metrics, compute_sge_metrics
from marin_dna.pipelines.rag_glm.hf_scoring import (
    RAG_COMPLETION_TOKENS,
    RAG_PREFIX_TOKENS,
    score_rag_completions_hf,
)

RAGBenchmark = Literal["mendelian_traits", "complex_traits", "sge"]
RAG_BENCHMARK_DATASETS: dict[RAGBenchmark, tuple[str, str]] = {
    "mendelian_traits": (
        "marin-dna/evals_mendelian_traits_rag_harness_255_v1",
        "9acedb683463477f34745af30a63a289873008a4",
    ),
    "complex_traits": (
        "marin-dna/evals_complex_traits_rag_harness_255_v1",
        "0252a883f650819a8e1fa22062027daafe956540",
    ),
    "sge": (
        "marin-dna/evals_sge_rag_harness_255_v1",
        "c20cc58fceb9bc053a55152a89d160f1b070f75d",
    ),
}
RAG_SCORE_COLUMNS: dict[RAGBenchmark, str] = {
    "mendelian_traits": "minus_llr_avg",
    "complex_traits": "abs_llr_avg",
    "sge": "minus_llr_avg",
}
VARIANT_KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]
DOCUMENT_SCORE_COLUMNS = ["ref_loglikelihood", "alt_loglikelihood", "llr"]
_BENCHMARK_METADATA: dict[RAGBenchmark, list[str]] = {
    "mendelian_traits": ["target", "subset", "match_group"],
    "complex_traits": ["label", "subset", "match_group"],
    "sge": ["label", "subset", "gene", "mavedb_urn"],
}


def rag_eval_columns(benchmark: RAGBenchmark) -> list[str]:
    """Return the minimal harness columns needed for scoring and metrics."""
    assert benchmark in RAG_BENCHMARK_DATASETS
    return [
        *VARIANT_KEY_COLUMNS,
        *_BENCHMARK_METADATA[benchmark],
        "variant_id",
        "document_id",
        "strand",
        "context",
        "ref_completion",
        "alt_completion",
    ]


def load_rag_eval_split(
    benchmark: RAGBenchmark,
    split: str,
    *,
    repo: str | None = None,
    revision: str | None = None,
) -> pl.DataFrame:
    """Read one immutable harness split directly from its pinned parquet."""
    assert split in {"train", "test"}
    assert repo is None or revision is not None, (
        "overriding the dataset repository also requires an explicit revision"
    )
    default_repo, default_revision = RAG_BENCHMARK_DATASETS[benchmark]
    resolved_repo = repo or default_repo
    resolved_revision = revision or default_revision
    assert resolved_repo
    assert len(resolved_revision) == 40, "dataset revision must be a full commit SHA"
    url = (
        f"https://huggingface.co/datasets/{resolved_repo}/resolve/"
        f"{resolved_revision}/{split}.parquet"
    )
    rows = pl.read_parquet(url, columns=rag_eval_columns(benchmark))
    assert rows.height > 0
    return rows


def nucleotide_token_ids(tokenizer: Any) -> Tensor:
    """Resolve the tokenizer's four unique A/C/G/T IDs without special tokens."""
    encoded = tokenizer(list("ACGT"), add_special_tokens=False)["input_ids"]
    assert len(encoded) == 4
    assert all(len(ids) == 1 for ids in encoded), "A/C/G/T must each be one token"
    token_ids = torch.tensor([ids[0] for ids in encoded], dtype=torch.long)
    assert token_ids.unique().numel() == 4
    return token_ids


def load_rag_tokenizer_hf(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None = None,
) -> Any:
    """Load the fixed RAG tokenizer without version-specific exporter metadata.

    Levanter currently emits a Transformers 5-only class name and special-token
    schema in ``tokenizer_config.json``. The immutable ``tokenizer.json`` is
    version-neutral, so reconstruct the experiment's known special-token
    contract explicitly instead of interpreting incompatible metadata.
    """
    model_path = Path(pretrained_model_name_or_path)
    if model_path.exists():
        assert model_path.is_dir()
        tokenizer_file = model_path / "tokenizer.json"
    else:
        tokenizer_file = Path(
            hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename="tokenizer.json",
                revision=revision,
            )
        )
    assert tokenizer_file.is_file()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="[BOS]",
        cls_token="[BOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=["[SEQ]"],
        clean_up_tokenization_spaces=False,
        model_max_length=2_048,
    )
    assert tokenizer.vocab_size == 8
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == tokenizer.cls_token_id == 2
    assert tokenizer.eos_token_id is None
    assert tokenizer.convert_tokens_to_ids("[SEQ]") == 3
    assert nucleotide_token_ids(tokenizer).tolist() == [4, 5, 6, 7]
    return tokenizer


def encode_rag_batch(
    tokenizer: Any, rows: pl.DataFrame
) -> tuple[Tensor, Tensor, Tensor]:
    """Tokenize one fixed-layout batch and assert its exact geometry."""
    required = {"context", "ref_completion", "alt_completion"}
    assert required <= set(rows.columns)
    assert rows.height > 0
    prefix = tokenizer(
        rows["context"].to_list(),
        add_special_tokens=True,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    ref = tokenizer(
        rows["ref_completion"].to_list(),
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    alt = tokenizer(
        rows["alt_completion"].to_list(),
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    assert prefix.shape == (rows.height, RAG_PREFIX_TOKENS)
    assert ref.shape == (rows.height, RAG_COMPLETION_TOKENS)
    assert alt.shape == ref.shape
    assert bool((ref[:, 1:] == alt[:, 1:]).all())
    assert bool((ref[:, 0] != alt[:, 0]).all())
    return prefix, ref, alt


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    parameters = getattr(model, "parameters", None)
    assert parameters is not None, "model has neither .device nor .parameters()"
    return next(parameters()).device


def score_rag_rows_hf(
    model: Any,
    tokenizer: Any,
    rows: pl.DataFrame,
    *,
    batch_size: int,
    device: str | torch.device | None = None,
) -> pl.DataFrame:
    """Add raw paired-completion log-likelihoods to materialized strand rows."""
    assert batch_size > 0
    required = {
        *VARIANT_KEY_COLUMNS,
        "variant_id",
        "document_id",
        "strand",
        "context",
        "ref_completion",
        "alt_completion",
    }
    assert required <= set(rows.columns)
    assert rows["document_id"].n_unique() == rows.height
    assert set(rows["strand"].unique()) == {"+", "-"}
    resolved_device = (
        torch.device(device) if device is not None else _model_device(model)
    )
    nucleotide_ids = nucleotide_token_ids(tokenizer).to(resolved_device)
    chunks: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, rows.height, batch_size):
            batch = rows.slice(start, batch_size)
            prefix, ref, alt = encode_rag_batch(tokenizer, batch)
            chunks.append(
                score_rag_completions_hf(
                    model,
                    prefix.to(resolved_device),
                    ref.to(resolved_device),
                    alt.to(resolved_device),
                    nucleotide_token_ids=nucleotide_ids,
                ).cpu()
            )
    scores = torch.cat(chunks, dim=0).float().numpy()
    assert scores.shape == (rows.height, 3)
    assert np.isfinite(scores).all()
    return rows.with_columns(
        *[
            pl.Series(column, scores[:, index], dtype=pl.Float32)
            for index, column in enumerate(DOCUMENT_SCORE_COLUMNS)
        ]
    )


def aggregate_rag_variant_scores(
    rows: pl.DataFrame, benchmark: RAGBenchmark
) -> pl.DataFrame:
    """Average raw strand atoms, then apply the benchmark score protocol."""
    required = {
        *VARIANT_KEY_COLUMNS,
        *_BENCHMARK_METADATA[benchmark],
        "variant_id",
        "strand",
        *DOCUMENT_SCORE_COLUMNS,
    }
    assert required <= set(rows.columns)
    group_columns = [*VARIANT_KEY_COLUMNS, "variant_id"]
    metadata_columns = _BENCHMARK_METADATA[benchmark]
    counts = rows.group_by(group_columns).agg(
        pl.len().alias("n_rows"),
        pl.col("strand").n_unique().alias("n_strands"),
        *[
            pl.col(column).n_unique().alias(f"n_{column}")
            for column in metadata_columns
        ],
    )
    inconsistency = (pl.col("n_rows") != 2) | (pl.col("n_strands") != 2)
    for column in metadata_columns:
        inconsistency |= pl.col(f"n_{column}") != 1
    assert counts.filter(inconsistency).is_empty(), (
        "each variant must have exactly two strands and consistent metric metadata"
    )

    forward = rows.filter(pl.col("strand") == "+").select(
        *group_columns,
        *metadata_columns,
        *[pl.col(column).alias(f"{column}_fwd") for column in DOCUMENT_SCORE_COLUMNS],
    )
    reverse = rows.filter(pl.col("strand") == "-").select(
        *group_columns,
        *[pl.col(column).alias(f"{column}_rc") for column in DOCUMENT_SCORE_COLUMNS],
    )
    assert forward.height == reverse.height == counts.height
    variants = forward.join(reverse, on=group_columns, how="inner", validate="1:1")
    variants = variants.with_columns(
        *[
            ((pl.col(f"{column}_fwd") + pl.col(f"{column}_rc")) / 2).alias(
                f"{column}_avg"
            )
            for column in DOCUMENT_SCORE_COLUMNS
        ]
    )
    assert np.allclose(
        variants["llr_avg"].to_numpy(),
        (
            variants["alt_loglikelihood_avg"] - variants["ref_loglikelihood_avg"]
        ).to_numpy(),
        rtol=1e-5,
        atol=1e-4,
    )
    score_column = RAG_SCORE_COLUMNS[benchmark]
    if score_column == "minus_llr_avg":
        variants = variants.with_columns((-pl.col("llr_avg")).alias(score_column))
    else:
        assert score_column == "abs_llr_avg"
        variants = variants.with_columns(pl.col("llr_avg").abs().alias(score_column))
    if "label" not in variants.columns:
        variants = variants.with_columns(pl.col("target").alias("label"))
    assert variants.filter(~pl.col(score_column).is_finite()).is_empty()
    return variants.sort(VARIANT_KEY_COLUMNS)


def compute_rag_benchmark_metrics(
    variants: pl.DataFrame,
    benchmark: RAGBenchmark,
    *,
    n_bootstrap: int = 1_000,
    rng: np.random.Generator | int | None = 0,
    n_min: int = 30,
) -> pl.DataFrame:
    """Compute the existing leaderboard metric on variant-level RAG scores."""
    score_column = RAG_SCORE_COLUMNS[benchmark]
    assert score_column in variants.columns
    scores = variants.select(score_column).to_pandas()
    if benchmark == "sge":
        dataset = variants.select("mavedb_urn", "gene", "subset", "label").to_pandas()
        metrics = compute_sge_metrics(
            dataset,
            scores,
            score_columns=[score_column],
            n_bootstrap=n_bootstrap,
            rng=rng,
            n_min_auprc=n_min,
        )
    else:
        dataset = variants.select("label", "subset", "match_group").to_pandas()
        metrics = compute_auprc_metrics(
            dataset,
            scores,
            score_columns=[score_column],
            n_bootstrap=n_bootstrap,
            rng=rng,
            n_min=n_min,
        )
    return pl.from_pandas(metrics)


def write_rag_evaluation_outputs(
    *,
    document_scores: pl.DataFrame,
    variant_scores: pl.DataFrame,
    metrics: pl.DataFrame,
    output_dir: str | Path,
    benchmark: RAGBenchmark,
    split: str,
    model: str,
    model_revision: str | None,
    dataset_repo: str,
    dataset_revision: str,
    code_revision: str,
    batch_size: int,
) -> None:
    """Write lossless score tables and a complete reproducibility manifest."""
    assert len(dataset_revision) == 40
    assert len(code_revision) == 40
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    document_scores.write_parquet(output / "documents.parquet", compression="zstd")
    variant_scores.write_parquet(output / "variants.parquet", compression="zstd")
    metrics.write_parquet(output / "metrics.parquet", compression="zstd")
    manifest = {
        "benchmark": benchmark,
        "split": split,
        "model": model,
        "model_revision": model_revision,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "code_revision": code_revision,
        "batch_size": batch_size,
        "n_document_rows": document_scores.height,
        "n_variants": variant_scores.height,
        "score_column": RAG_SCORE_COLUMNS[benchmark],
        "aggregation": "average raw fwd/rc LLR, then apply score transform",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
